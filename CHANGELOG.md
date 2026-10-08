# Changelog

## Unreleased

- Bench: `APA_FULL=1` also scores the `APA_PAGED` and `APA_CHUNKS` outputs over all pairs (`  full <path>` lines).
- Paper tables eps / context length on all 49152 pairs (AUDIT.md, Phase 3). lc_122880_2 min cos at eps 0.001 /
  0.003 / 0.01 / 0.02: 0.926464 / 0.857105 / 0.834358 / 0.624999 (sample: 0.967307 / 0.946039 / 0.944326 /
  0.944326). Paged = flat on all pairs. Incremental tile cache, lc_122880_2, eps 0.005: min 0.679235, 275 pairs
  < 0.9 (full requantization 0.841302, 14).
- Incremental tile cache: K mean and head scales rebuilt when the context grows by `APA_KV_RESTAT` 1.125 (was 2).
  Cause: frozen K mean (AUDIT.md, Phase 4). lc_122880_2, eps 0.005, 60 chunks: last chunk min cos 0.679235 ->
  0.829814, pairs < 0.9 275 -> 17 (full requantization 0.841302, 14); worst chunk vs full requantization, pairs
  < 0.99: 1452 -> 29; 178.32 -> 182.88 ms (full requantization 196.59 ms).
- Bench: `APA_CHUNKS=2` compares every chunk with a full requantization; diagnostics `-DAPA_DBG_KMEAN_LEN`,
  `-DAPA_DBG_KMEAN_FROM`.
- Pass 2: one stream of the union of a q block's hot tiles through 3 K/V smem stages, shared by its 12 warps (was 3
  independent 4-warp groups with bar.sync per tile). Outputs bitwise equal; attention at eps 0.005 / 0.01, mean of 3:
  -1.5 / -1.7 % (lc_122880_0), -2.9 / -1.6 % (lc_122880_1), -2.4 / -1.2 % (lc_122880_2), -2.5 / -3.1 % (lc_32768_1);
  all-exact +2.9 to +3.5 %. Bench: `p2load` (12-warp union) replaces `p2cta`; `APA_UNION=1`, `APA_OUT=file`.

## 0.3.0 (2026-10-08)

- Pass 1 cold frame: a hot tile lifts the running max of the cold accumulation to at most its own max - 32
  (log2, `APA_HOT_DROP`). Before, a hot sink set the frame; cold 16-key groups more than 2^-10 below it hit
  the -6 scale clamp and became E2M1 zero (AUDIT.md, Phase 1). All 49152 pairs, eps 0.005, min cos
  lc_122880_{0,1,2}, lc_32768_{0,1,2}: 0.634875 / 0.925502 / -0.638155 / 0.757021 / 0.927160 / -0.594803 ->
  0.959676 / 0.975775 / 0.841302 / 0.927517 / 0.980919 / 0.969776; pairs < 0: 789 (lc_122880_2) and 110
  (lc_32768_2) -> 0. Speed equal within noise (same run); paged path same accuracy; incremental (512-pair
  sample) min -0.431078 -> 0.822568; deterministic.
- Bulk copy into `shared::cta` (was `shared::cluster`): no `__cuda_syscall_cp_async_bulk_unicast`, first launch
  no longer raises the stack limit to 14448 B/thread (about 3.5 GB local memory).
- `APA_P2_F32O=1` (default 0): pass-2 O in fp32 across tiles, f16 per 16-token k-step. All-exact lc_122880_1
  (512-pair sample) cos 0.999634 -> 1.000000, min 0.988660 -> 0.999997 (12.431 -> 17.339 ms); attn at eps 0.005
  +14.5 % (lc_122880_1) and +24.9 % (lc_32768_1), register spills, cos 0.999791 -> 0.999804.
- Bench: `APA_FULL=1` (all pairs vs FP32, histogram, relL2, top-20), `APA_REF_CACHE`, `-DAPA_DBG` pass-1 row
  export, `APA_DIAG=1` (worst sampled row). Comment fixes: pass 2 accumulates in f16; stray `#pragma unroll`.

## 0.2.0 (2026-10-08)

- Deterministic: K column sums per 16-tile chunk in a fixed order (registers, xor shuffles, warps in order,
  chunks summed in order in `finalize_stats_kernel`) instead of shared and global float atomics. 5 reruns of
  prep + attention on lc_122880_{0,1,2}: 539072 / 1329742 / 950802 differing output elements before, 0 now.
- Prep faster: 0.865 -> 0.731 ms at kv 122880, 0.244 -> 0.163 ms at kv 32768 (lc_122880_1, eps 0.005).
  Attention time and cos unchanged.
- `cudaFuncSetAttribute` once per kernel instance and device instead of every launch.
- Workspace: `kpart` ([B*Hkv][ceil(ntkv / 16)][D] floats) at the end; `workspace_bytes` grows accordingly.
- `APA_VERSION_MAJOR/MINOR/PATCH` in `apa.cuh`; `APA_DET=1` bench mode.

## 0.1.0 (2026-10-08)

- First release as its own repository (from kekzl/ra2 `src/apa`, 85f508e), MPL-2.0: pass 1 all-FP4, pass 2
  exact FP16 over hot tiles, log-sum-exp merge, one fused launch; flat, paged FP16 + flat tail, chunked
  prefill with a per-layer tile cache (`KvState`); pass-1 tile order sink first, then the diagonal backwards.
