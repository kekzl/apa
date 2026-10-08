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

## Phase 1: worst rows, full mode

### Tooling

| Item | Where |
|---|---|
| `APA_FULL=1` | bench: all 49152 pairs vs unchanged `ref_kernel` (batches of 1024, `APA_REF_CACHE`); pooled cos, mean of per-pair cos, min, counts < 0.99 / 0.9 / 0, relL2 mean / p50 / p99 / p999 / max, cos histogram, top-20 pairs |
| Top-20 columns | warp, q block, hot tiles of the warp, merge vs cold-only; FP32: cold mass, max tile, max cold tile, cold tiles with FP32 share > eps, n90, cold-part norm / ref norm; K / Q channel amax spread of the pair's heads |
| `-DAPA_DBG` | pass-1 export per row: m, lambda, l_cold (ml units), max tile share at test time; default build unchanged |
| Builds | PERF_LOG "Phase 1 full mode"; all with the shared::cta fix (d6aecf3) |

DBG column "fp4 max tile" is 1.0000 in every row: the first visited tile (sink) is tested against lambda = 0.
The column carries no information; hypothesis (c) is judged from the FP32 columns instead.

### Sample vs all pairs (eps 0.005, same run, PERF_LOG `ab_*`)

| Dump | 0.2.0 sampled min | 0.2.0 all-pairs min | pairs < 0 | < 0.9 | 0.3.0 sampled min | 0.3.0 all-pairs min | < 0 | < 0.9 |
|---|---|---|---|---|---|---|---|---|
| lc_122880_0 | 0.898838 | 0.634875 | 0 | 319 | 0.969733 | 0.959676 | 0 | 0 |
| lc_122880_1 | 0.970893 | 0.925502 | 0 | 0 | 0.997192 | 0.975775 | 0 | 0 |
| lc_122880_2 | -0.444615 | -0.638155 | 789 | 5460 | 0.944326 | 0.841302 | 0 | 14 |
| lc_32768_0 | 0.879942 | 0.757021 | 0 | 50 | 0.967587 | 0.927517 | 0 | 0 |
| lc_32768_1 | 0.984021 | 0.927160 | 0 | 0 | 0.996628 | 0.980919 | 0 | 0 |
| lc_32768_2 | 0.100775 | -0.594803 | 110 | 5655 | 0.987198 | 0.969776 | 0 | 0 |

The 512-pair sample overstates the minimum on every dump. Mean of per-pair cos, 0.2.0 -> 0.3.0:
0.995494 -> 0.998512, 0.999064 -> 0.999717, 0.931721 -> 0.998438, 0.997637 -> 0.999037, 0.999340 -> 0.999827,
0.955446 -> 0.999372. Attn ms 0.2.0 / 0.3.0 same run: 4.555 / 4.490, 4.676 / 4.759, 4.758 / 4.840, 2.245 / 2.285,
1.881 / 1.889, 1.856 / 1.863. Release (rel*) and DBG (d*) builds give identical accuracy lines.

### Diagnostic runs, lc_122880_2, all pairs, eps 0.005 unless noted

| Run | mean cos | min | < 0.9 | < 0 |
|---|---|---|---|---|
| 0.2.0 default | 0.931721 | -0.638155 | 5460 | 789 |
| 0.2.0 `-DAPA_EXACT_MAX` | 0.991244 | 0.251078 | 960 | 0 |
| 0.2.0 `APA_ORDER=0` | 0.934156 | -0.638155 | 5115 | 787 |
| pure FP4, eps 1e9 (0.2.0 = 0.3.0, no hot tile) | 0.868772 | -0.346474 | 16899 | 25 |
| all exact, eps < 0 (0.2.0 = 0.3.0, pass 1 skipped) | 0.993600 | 0.834726 | 465 | 0 |
| 0.3.0 default | 0.998438 | 0.841302 | 14 | 0 |
| 0.3.0 `-DAPA_EXACT_MAX` | 0.998437 | 0.846343 | 14 | 0 |
| 0.3.0 `APA_ORDER=0` | 0.998575 | 0.842157 | 14 | 0 |

0.2.0 at eps 0.005 has 789 pairs < 0, pure FP4 has 25: adding exact hot tiles made those rows worse. All 20
worst pairs of every 0.005 log are on the merge path (warp with hot tiles); none is cold-only.

### Hypotheses

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| (a) | diffuse rows, accumulated FP4 error over ~1900 cold tiles | residual cause in 0.3.0, not the 0.2.0 failure | worst 0.3.0 pairs: max cold tile 0.0009 to 0.0016, FP32 cold mass 0.0548 to 0.0903, cold-part norm / ref norm 1.859 to 2.330 (hot and cold parts cancel); FP4 error of the tail is amplified ~2x |
| (b) | E2M1 truncation, systematic mass loss | confirmed (0.2.0) | lc_122880_2 0.2.0 top 8: FP32 cold mass 0.0962 to 0.1511, FP4 cold share l_cold / lambda 0.0069 to 0.0254, m 21.94 to 22.68 (sink frame); 0.3.0 same rows' class: FP4 0.0462 to 0.0977 vs FP32 0.0548 to 0.0903, m 9.12 to 12.78 |
| (c) | hot test on FP4 scores misses a hot tile | occurs, not the cause | lc_122880_0 / _2 top-20: 0 cold tiles with FP32 share > eps; lc_122880_1 0.3.0: 7 of 20 pairs, up to 8 tiles, max 0.0118; those pairs cos >= 0.975775 |
| (d) | merge / normalisation, l_cold ~ 0, -6 clamp | clamp confirmed as the mechanism of (b); merge correct | `-DAPA_EXACT_MAX` (TAU 4 -> 0) moves the unclamped range of a group from 2^-10 to 2^-14 below m and turns 789 pairs < 0 into 0 (min 0.251078); l_cold 0.007 to 0.026 (not 0, no fp32 underflow) |
| (e) | Q / K outlier channels vs global head scale | rejected | channel amax spread (max / median) all heads, lc_122880_2: K 2.0 to 4.4, Q 2.9 to 10.8; worst heads 18 to 20 (kv head 6): K 3.2, Q 3.7 to 4.1; heads with Q spread 10.8 / 9.0 / 8.8 not among the worst |

Cause: in 0.2.0 the pass-1 frame m followed every tile, so the hot sink (FP32 max tile share 0.8365 to 0.8970, top 8) set m. The
P scale exponent of a 16-key group is clamped at -6; a group more than 2^-10 below m quantizes to E2M1 zero.
The tail (FP32 cold mass ~0.10, max cold tile <= 0.0025) vanished from the cold partial, while its output
component exceeds the reference norm (cancellation), so the merged row lost its direction.

Fix: 0.3.0 cold frame (`APA_HOT_DROP` 32, commit 6801a50, made before this audit from a single sampled row);
the all-pairs runs above confirm it. Tile order and exact max are not fixes: `APA_ORDER=0` leaves 787 pairs < 0
in 0.2.0 and costs hot share 11.0 vs 6.0 % (attn 6.207 vs 4.995 ms, lc_122880_2, d32 builds) in 0.3.0.

Guarantee (precise): a hot tile no longer moves the cold frame above (its max - 32) in log2. A cold 16-key
group is quantized without the -6 clamp if its max is at most 10 (TAU 4) below the cold frame. So a hot tile can
push a cold group to E2M1 zero only if that group lies more than 2^-42 below the hot tile's max. Nothing bounds
the FP4 error of the remaining cold mass; the hot test bounds the FP4-estimated share of each cold tile only.

### Claims to correct (done in phase 2, see below)

| Claim | Where | Correction |
|---|---|---|
| "worst-case cosine ... from -0.444615 to 0.944326" | paper abstract, 4.2, CHANGELOG 0.3.0, README | sample minima; all pairs: -0.638155 -> 0.841302 (lc_122880_2) |
| cos min columns | README table, paper Tables 3 / 4 / 5 | label as 512-pair sample or replace with all-pairs values |
| "cos mean" | README, paper | pooled cosine; say so or add mean of per-pair cos |
| 2.50x / 2.32x | paper, README 0.2.0 text | attention only, prep excluded; with prep 2.196 / 2.006 |
| apa_pass2.cuh:6 "f16 x f16 -> f32" | code comment | fixed in phase 1: "f16 accumulate" |
| Proposition | paper 3.2 | holds for FP4 estimates of l_t and lambda |
| PPL table "ppl_corpus_45k" | README, paper Table 7 | name the file: ppl_corpus_45k_gemma4_turn.txt (chat-turn wrapped), differs from imp ppl_corpus_45k.txt (plain); with chunk 2048 and apa_min_kv 8192 APA only runs on chunks with kv >= 8192 |

## Phase 2: claim corrections (README, CHANGELOG, paper)

| Claim | Change |
|---|---|
| worst-case cos 0.944326 / -0.444615 | all-pairs minima everywhere (0.841302 / -0.638155); sample minima kept only where labelled |
| cos columns | README: pooled cos, mean cos, min cos, pairs < 0.9 from `APA_FULL`; paper Tables 4 / 5 labelled sample mode |
| speedups | paper: attention kernel only, prep excluded; with prep 2.196x at 122880 keys |
| Proposition, contributions | stated for FP4 estimates; FP32 cold-tile share up to 0.0118 at eps 0.005 |
| cold-frame guarantee | paper 3.2: 2^-42 bound, no bound on remaining cold FP4 error |
| "with delta = 0 bitwise 0.2.0" | not measured bitwise; now "reproduces the 0.2.0 metrics to all six printed digits" |
| "probabilities near 2^-20 of the sink" | was an estimate; replaced by measured "largest cold tile <= 0.0025" |
| PPL corpus | both files named; imp 0.3.0 numbers (relayed, PERF_LOG) added with their setup |

## Phase 3: sample-mode tables and paths on all pairs

Runs: PERF_LOG "Phase 3 full mode". Accuracy and hot share are deterministic: every sample-mode cos / min / hot of the
old Tables 4 / 5 reproduced exactly, so their times stay and the accuracy columns are replaced.

| Item | Sample | All pairs | Change |
|---|---|---|---|
| Table 4, lc_122880_2 min, eps 0.001 / 0.003 / 0.01 / 0.02 | 0.967307 / 0.946039 / 0.944326 / 0.944326 | 0.926464 / 0.857105 / 0.834358 / 0.624999; < 0.9: 0 / 11 / 15 / 18 | paper Table 4 and text |
| Table 4, lc_122880_1 / lc_32768_2 min over eps | >= 0.988660 / >= 0.981491 | >= 0.930447 / >= 0.928616, 0 pairs < 0.9 | paper Table 4 |
| Table 5, APA min, 8192 .. 122880 keys | 0.996107 / 0.994661 / 0.988885 / 0.990023 / 0.997192 | 0.963431 / 0.950786 / 0.963669 / 0.949406 / 0.975775 | paper Table 5 (+ exact min) |
| Paged (bs 16) vs flat, lc_122880_{1,2} | equal | equal to six digits | paper text |
| Incremental vs full requant, lc_122880_2 | 0.822568 vs 0.944326 | 0.679235 / 275 < 0.9 vs 0.841302 / 14 | paper text, Limitations, README |
| Incremental vs full requant, lc_122880_1 | 0.993415 vs 0.997192 | 0.974581 vs 0.975775, 0 < 0.9 | paper text |

Open: 14 pairs < 0.9 (lc_122880_2, eps 0.005), 18 at eps 0.02; incremental cache adds 261 on the same dump.

## Phase 4: incremental tile cache

Runs: PERF_LOG "Phase 4". lc_122880_2, eps 0.005, all pairs unless noted.

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| (a) | KV_HEADROOM 4 pushes UE4M3 block scales into the subnormal range | rejected | headroom 1 / 2 / 4: bitwise equal results on 3 dumps (power-of-two shift); headroom 3 differs (macro active); headroom 1 does not clip, so the first 65536 keys hold the maxima |
| (b) | frozen K mean (stats from the last restat, 65536 keys at the last chunk) | confirmed | full requantization of 120832 keys with the K mean over the first 65536 keys: min 0.559054, 431 pairs < 0.9 = incremental with restat 2 (0.559054, 430) |

K-mean age (kv 120832, mean over first L keys) vs pairs < 0.9: L 65536 / 81920 / 98304 / 108544 / 120832 -> 431 / 222 / 104 / 56 / 24
(lc_122880_0: 0 throughout, min 0.966430 .. 0.954930). K mean over recent keys only ([4096, Skv) / [60416, Skv)): lc_122880_2
12 / 0 pairs < 0.9, lc_122880_0 cos < 0.99 1189 / 2124 vs 1142: no consistent gain, full requantization unchanged.

| KV_RESTAT | last chunk min / < 0.9 | worst chunk vs requant, max cos < 0.99 (lc_122880_2 / _0 / _1) | incremental ms |
|---|---|---|---|
| 2 (0.3.0) | 0.679235 / 275 | 1452 / 33 / 10 | 178.32 |
| 1.5 | 0.745347 / 130 | 610 / 5 / 9 | 179.98 |
| 1.25 | 0.841302 / 14 (restat at the last chunk) | 59 / 2 / 6 | 180.99 |
| 1.125 (new default) | 0.829814 / 17 | 29 / 2 / 6 | 182.88 |
| full requant | 0.841302 / 14 | - | 196.59 |

Decision: KV_RESTAT 1.125. Open: per-restat K mean with a per-row score offset in pass 1 / pass 2 (removes the age effect).

## Phase 5: test matrix

Runs: PERF_LOG "Phase 5". Reusable: `bench/test.sh` (12 dumps x 9 checks + 7 builds; last run 115 ok, `ALL OK`).

| Check | Scope | Result |
|---|---|---|
| builds | default + 11 macro variants | 0 errors; 1 warning (division by zero, APA_PREP_SAMPLE=0) fixed |
| regression, bitwise | APA_PREP_SAMPLE=0 vs before 516ce53: 12 dumps x eps -1 / 0.005, APA_KV 2048 .. 65537 | all equal |
| determinism | 12 dumps, 5 reruns | 0 elements differ |
| batch 2 (batch 1 = V negated) | 12 dumps, APA_KV 2048 .. 65537 | bitwise; only +0 vs -0 from exact cancellation |
| paged vs flat, bitwise | block size 1 / 16 / 48 / 64 / 256, unshuffled, 12 dumps | equal |
| chunked vs prefill, bitwise | every chunk, 12 dumps | equal at every restat (27 of 60 at 122880 keys) |
| prep overflow redo (forced) | 12 dumps | bitwise equal to exact stats |
| KvState overflow redo (forced) | 4 dumps | redo chunks bitwise equal to prefill |
| all pairs, eps 0.005 | 12 dumps | no NaN, no cos < 0; lc_65536_0: 124 pairs < 0.9, min 0.773144 |
| compute-sanitizer | memcheck / racecheck / synccheck / initcheck | not run: WSL2 needs EnableDebuggerInterface.bat (admin) |

| Defect | Fix |
|---|---|
| prefill_paged returned false for tail 0 (empty pool, first chunk): block table required but never read | accepted as flat; `__builtin_ctz(0)` guarded |
| KvState used exact unrounded stats while prefill sampled / rounded: chunk 0 already differed (min cos 0.777086, lc_122880_2) | same sampling and scales, overflow redo; restats bitwise equal |
| KvState: new keys beyond the frozen head scales saturated silently | overflow flag + gated redo |
| bench: APA_B2 with APA_KV, stale-cache test with 1 chunk (exit 6) | fixed (bench only) |

Open: lc_65536_0 (124 pairs < 0.9 at eps 0.005, 0.3.0 prep: 149) is the worst dump measured; cause not
measured yet (`-DAPA_DBG` top-20 as in Phase 1).
