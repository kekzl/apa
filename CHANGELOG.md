# Changelog

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
