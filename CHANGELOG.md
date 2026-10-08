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
  all-exact +2.9 to +3.5 %; paged prep+attn 5.076 ms (flat 4.906). Bench: `p2load` (12-warp union) replaces
  `p2cta`; `APA_UNION=1`, `APA_OUT=file`.
- Prep: stats over every s-th 1024-key chunk (>= 16 chunks, `APA_PREP_SAMPLE`, 0 = all); sampled head scales x4;
  head scales rounded up to powers of two. A 16-block beyond the UE4M3 range sets `Workspace::ovf` and reruns
  exact stats + KV quant on the device (forced test: bitwise equal to exact power-of-two stats). Prep 0.733 -> 0.482 ms
  (lc_122880_1), 0.162 -> 0.138 ms (lc_32768_0). All pairs, eps 0.005, mean cos
  exact -> sampled: 0.998512 / 0.999717 / 0.998438 / 0.999037 / 0.999827 / 0.999372 -> 0.998536 / 0.999705 /
  0.998352 / 0.999046 / 0.999837 / 0.999420; min 0.959676 / 0.975775 / 0.841302 / 0.927517 / 0.980919 / 0.969776 ->
  0.957087 / 0.972723 / 0.847877 / 0.938131 / 0.990549 / 0.970655; pairs < 0.9: 14 -> 13 (lc_122880_2).
- KvState (prefill_incremental): same sampled stats and power-of-two head scales x4 as prep; new tiles beyond the
  frozen scales set the overflow flag and redo exact stats + every tile on the device (before: saturated). Output
  at every restat bitwise equal to prefill (lc_122880_*: 27 of 60 chunks, lc_32768_*: 12 of 16; before: chunk 0
  only); worst chunk vs prefill, lc_122880_2: min cos 0.777086 -> 0.888330. prep: head scales x4 also unsampled
  (output bitwise unchanged on 12 dumps). APA_PREP_SAMPLE=0 keeps the 0.3.0 prep and KvState bitwise.
- prefill_paged accepts tail == 0 (empty pool, first chunk) without a block table; it returned false.
- Bench: `APA_B2=1` (batch 2, V negated: bitwise checks), paged vs flat and stale cache counted bitwise,
  `APA_CHUNKS=2` counts bitwise equal chunks; `bench/test.sh` runs the test matrix (AUDIT.md, Phase 5).

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
