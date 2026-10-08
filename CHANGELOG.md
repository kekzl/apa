# Changelog

## 0.3.0 (2026-10-08)

- Pass 1 cold frame: a hot tile lifts the running max of the cold accumulation to at most its own max - 32
  (log2, `APA_HOT_DROP`). Before, a hot sink set the frame and cold P fell below the UE4M3 scale floor
  (2^-6) to E2M1 zero: the worst row (lc_122880_2 s 190 h 20) lost its whole cold part (|out| 0.2531 =
  |hot pv| 0.2527, |ref| 0.1670). eps 0.005, cos min over 6 dumps: -0.444615 / 0.100775 / 0.879942 /
  0.898838 / 0.970893 / 0.984021 -> 0.944326 / 0.987198 / 0.967587 / 0.969733 / 0.997192 / 0.996628.
  Speed equal within noise; paged path same accuracy; incremental min -0.431078 -> 0.822568; deterministic.
- `APA_P2_F32O=1` (default 0): pass-2 O in fp32 across tiles, f16 per 16-token k-step. All-exact cos
  lc_122880_0 0.996468 -> 0.999998, lc_122880_2 min 0.876635 -> 0.999939; attn +8 to 25 % at eps 0.005
  (register spills) with no accuracy gain there.
- Bench: `APA_DIAG=1` prints the worst sampled row (norms, V cancellation, hot / cold mass).
- imp perplexity and prefill numbers in the README are 0.2.0 (not yet re-measured).

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
