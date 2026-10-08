# AUDIT

Claims in README.md and paper/apa.tex versus code and measurements. Branch `apa-cold-frame`, base 4963e56.
Measurements: PERF_LOG.md (append-only). One section per phase.

## Phase 0: audit (no edits)

### Metrics (bench/bench_apa.cu)

| Item | Code | Finding |
|---|---|---|
| Sample | `ns = 512`, `std::mt19937 rng(7)`, `(rng() % n, rng() % nh)` (l. 100-103) | 512 of n * nh = 2048 * 24 = 49152 (row, head) pairs (1.04 %), with replacement, fixed seed |
| Reference | `ref_kernel` (l. 26-72): fp32 dot products of FP16 Q/K, `expf`, fp32 sums | FP32 reference on the sampled pairs only |
| "cos mean" | `dot / sqrt(na * nb)` over all 512 pairs (l. 118-131) | pooled cosine of the concatenated vectors, not the mean of per-pair cosines; large-norm rows weigh more |
| "cos min" | min over the 512 per-pair cosines | worst of the sample, not of all pairs |
| Hot share | set bits over active (warp, tile) pairs (l. 170-183) | per (warp, tile); pass-2 load share is the 4-warp union (`p2cta`), larger |
| attn ms | `time()`: 1 warm-up, then min over 5 reps of (5 launches / 5) (l. 134-147) | prep excluded; `prep ms` timed separately |
| TOPS | `4 * nh * sum_i min(off + i + 1, kv) * hd / t` | causal pairs only |

### Headline speedups

| Claim | Where | Numerator / denominator | Prep | With prep |
|---|---|---|---|---|
| 2.32x | README 0.2.0, paper 545d42f | 10.725 / 4.615 ms, lc_122880_1 | excluded | 10.725 / (4.615 + 0.731) = 2.006 |
| 2.50x | paper 4963e56 Table 5 | 13.023 / 5.200 ms, lc_122880_1 | excluded | 13.023 / (5.200 + 0.730) = 2.196 |

- Denominator "every tile exact" = `attn(eps < 0)`: `n1 = 0`, no pass-1 CTAs; all masks 0xFF, `ml = 0`;
  pass 2 alone (f16 HMMA, f16 O) over every tile (include/apa/apa.cuh:189-200). It is APA's own pass 2,
  not FlashAttention-2. It still reads `ksum` from prep (row shift); its time excludes prep as well.
- Neither ratio is labelled as attention-only in the paper. To fix in the paper: state "attention kernel
  time, prep excluded" and give the prep-inclusive ratio.

### Pass-2 accumulation

| Location | Says | Code |
|---|---|---|
| include/apa/apa_pass2.cuh:6 | `QK^T and PV on mma.sync m16n8k16 (f16 x f16 -> f32)` | `mma_f16h` (l. 14-19): `.f16.f16.f16.f16`, f16 accumulate, for QK^T (`qk16`) and PV (`OAcc::mma2`) |
| include/apa/apa_pass2.cuh:14 | `f16 x f16 -> f16 accumulate (full-rate HMMA on sm_120, as imp's FA2 default)` | matches code |

Contradiction: line 6 is wrong. Default `APA_P2_F32O=0` keeps O in half2 across tiles; `=1` sums each
16-token k-step in f16 and O in fp32. S = QK^T is f16-accumulated in both. Paper Section 3.3 says
"f16 inputs, f16 accumulation" (correct).

### Claims beyond the measurement

| Claim | Status |
|---|---|
| Proposition (paper 3.2): cold tile `l_t <= eps * lambda_inf` | holds for the FP4 estimates `l_t`, `lambda` that pass 1 computes; says nothing about the FP32 mass share (hypothesis (c) below) |
| "monotone safe" (contributions list) | same restriction |
| cos mean / min "on 512 sampled pairs" | min is the sample minimum; worst pair over all 49152 unknown |
| imp prefill (6661.95 / 5241.18 / 11659.35 / 6433.23 ms) and perplexity table | measured in imp with APA 0.2.0 / 0.1.0; no log in this repo; not reproducible from here |
| "exact arithmetic on 6 to 10 % of the tiles" (conclusion) | (warp, tile) compute share; pass-2 loads cover the 4-warp union (lc_122880_*: 11.4 to 18.0 % in PERF_LOG) |
| Determinism "bitwise identical" | 5 reruns, same binary, same GPU; nothing across GPUs or builds |
| Times | drift up to 10 % between runs (lc_122880_1 eps 0.005: 4.707 and 5.200 ms); only within-run comparisons hold |

### State of the worst-row issue at Phase 0 start

- 0.3.0 (6801a50) already changes pass 1 (`APA_HOT_DROP`, cold frame) after a single-row diagnosis
  (`APA_DIAG=1`, worst of 512 samples). Phase 1 re-checks the cause on all pairs with the A/B switch
  `-DAPA_HOT_DROP=0` (= 0.2.0 pass 1) and keeps or revises the fix based on that.
