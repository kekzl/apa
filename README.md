# APA: adaptive-precision attention

Causal flash-attention prefill for RTX 5090 (sm_120a), header-only CUDA. Long-context attention mass sits in
few KV tiles; APA computes those exactly and the rest in FP4.

| Step | What |
|---|---|
| Prep | K mean and head scales (sampled stats); Q (two terms), K, V to E2M1 with one UE4M3 scale per 16 values: the code in nearest(amax / 6) + {0, -1, 1, -2, 3 .. 6} with the least squared error (f16x2 ranking) |
| Pass 1 | all-FP4 flash attention (`mma.sync kind::mxf4nvf4`) over all 64-key tiles, Q as two E2M1 terms (term 2 = residual of term 1), P scales per 16 keys in UE4M3; a tile whose share of the running row sum exceeds `eps` is hot: no P·V here, bit in a per-warp mask |
| Pass 2 | exact FP16 attention (`mma.sync m16n8k16`) over the hot tiles only; one K/V tile stream per q block shared by its 12 warps |
| Merge | log-sum-exp of the cold-tile partials (pass 1) and the hot result (pass 2) |
| Launch | one fused kernel: pass-1 CTAs, then persistent pass-2 workers on ticket counters and per-q-block ready flags |

Scope: head dim 128, causal (`q_offset` = position of Q row 0), GQA (`H % Hkv == 0`), FP16 Q/K/V. K/V flat, or a
paged FP16 pool through a block table plus a flat tail (chunked prefill). Pass-1 tile order: sink tile first,
then the diagonal backwards.

## Results

Standalone, Llama-3.2-3B attention dumps (2048 queries at kv 122880, 65536 or 32768, 24 / 8 heads), eps 0.005, every
(row, head) pair (49152) vs FP32 (`APA_FULL=1`), one run (bench/test.sh run; PERF_LOG.md, "Phase 10"):

| Dump | pooled cos | mean cos | min cos | pairs < 0.9 | pairs < 0.99 | hot tiles | attn ms | prep ms |
|---|---|---|---|---|---|---|---|---|
| lc_122880_0 | 0.999794 | 0.999117 | 0.984284 | 0 | 887 | 10.1 % | 5.329 | 0.503 |
| lc_122880_1 | 0.999973 | 0.999959 | 0.997452 | 0 | 0 | 9.5 % | 5.558 | 0.497 |
| lc_122880_2 | 0.999927 | 0.999769 | 0.986760 | 0 | 4 | 5.9 % | 5.776 | 0.499 |
| lc_65536_0 | 0.999849 | 0.999391 | 0.987245 | 0 | 14 | 19.2 % | 3.424 | 0.294 |
| lc_65536_1 | 0.999980 | 0.999968 | 0.997996 | 0 | 0 | 13.7 % | 3.319 | 0.288 |
| lc_65536_2 | 0.999940 | 0.999828 | 0.992051 | 0 | 0 | 9.4 % | 3.429 | 0.293 |
| lc_32768_0 | 0.999915 | 0.999719 | 0.993301 | 0 | 0 | 39.0 % | 2.597 | 0.178 |
| lc_32768_1 | 0.999987 | 0.999974 | 0.998725 | 0 | 0 | 23.2 % | 2.048 | 0.180 |
| lc_32768_2 | 0.999962 | 0.999889 | 0.992835 | 0 | 0 | 18.2 % | 2.099 | 0.182 |

0.4.0, same dumps: 137 pairs < 0.9 (lc_122880_2: 13, lc_65536_0: 124), 5196 pairs < 0.99 (12 dumps; now 905), min cos
0.773144 (lc_65536_0; now 0.984284); attn +6.7 to +25.4 % (0.6.0 vs 0.4.0, same run, min of 2 reps; paper, "Error
source of the worst rows").

| Term | Definition |
|---|---|
| pooled cos | cosine of all pairs' outputs concatenated (the bench's default `cos`, there over 512 sampled pairs) |
| mean cos / min cos | mean / minimum of the 49152 per-pair cosines |
| hot tiles | share of active (warp, tile) pairs sent to pass 2 (bench `hot`; `p2load`: tiles pass 2 loads per q block) |
| attn ms | fused attention kernel, prep excluded; min over 5 reps of 5 launches; drifts up to 10 % between runs |

APA 0.2.0 (0.3.0 audit run, PERF_LOG `ab_*`): min cos 0.634875 / 0.925502 / -0.638155 / 0.757021 / 0.927160 / -0.594803, pairs < 0:
0 / 0 / 789 / 0 / 0 / 110, attn 4.555 / 4.676 / 4.758 / 2.245 / 1.881 / 1.856 ms (cause: AUDIT.md, Phase 1).

APA 0.3.0 with every tile exact (`eps < 0`, pass 2 alone, f16 O), all pairs: lc_122880_1 mean cos 0.999505, min
0.974166; lc_122880_2 mean 0.993600, min 0.834726. `-DAPA_P2_F32O=1` (512-pair sample): lc_122880_1 cos
1.000000, min 0.999997, 12.431 -> 17.339 ms. Deterministic: 5 reruns of prep + attention give bitwise identical
outputs (`APA_DET=1`).

In [imp](https://github.com/kekzl/imp) (`attention.apa_eps`), RTX 5090, `apa_min_kv 8192`; measured by the imp
integration (relayed, PERF_LOG.md). APA 0.5.0 and 0.6.0: kekzl/imp#2648 (imp base 2934f575), one run; FA2 and 0.4.0:
kekzl/imp#2646. Llama-3.2-3B Q8_0, 106451-token prompt, `--max-seq-len 108000`, prefill ms, rep 1 / rep 2:

| Attention | eps | default (sparse prefill) | dense (`sparse_prefill_topk_tokens=0`) |
|---|---|---|---|
| FA2 (`apa_eps 0`) | | 7180.71 / 7025.34 | 11884.58 / 11902.36 |
| APA 0.4.0 | 0.01 | 5094.99 / 4848.68 | 6220.93 / 6227.74 |
| APA 0.5.0 | 0.01 | 4946.49 / 5005.64 | 6703.86 / 6669.97 |
| APA 0.6.0 | 0.01 | 5056.78 / 5029.35 | 6730.54 / 6874.74 |
| APA 0.5.0 | 0.005 | 5242.15 / 5477.81 | 7705.54 / 7685.13 |
| APA 0.6.0 | 0.005 | 5594.75 / 5738.88 | 7763.12 / 7725.78 |

Perplexity, corpus `ppl_corpus_45k_gemma4_turn.txt` (below), Qwen3-8B Q8_0 (14034 tokens) / Llama-3.2-3B Q8_0
(13303 tokens): FA2 10.7974 / 17.3644.

| APA | eps 0.01 | eps 0.005 |
|---|---|---|
| 0.4.0 | 10.8063 / 17.3599 | |
| 0.5.0 | 10.7972 / 17.3679 | 10.8071 / 17.3663 |
| 0.6.0 | 10.7971 / 17.3704 | 10.7965 / 17.3655 |

APA e71624b (0.3.0), default sparse prefill, `apa_min_kv 8192`, mean of 2, prefill ms:

| Model | Prompt tokens | FA2 (`apa_eps 0`) | APA eps 0.01 |
|---|---|---|---|
| Llama-3.2-3B | 106451 | 6746.01 | 4824.90 |
| Qwen3-4B-2507 | 112280 | 8802.52 | 6969.61 |
| Qwen3-8B | 33727 | 3227.40 | 2848.55 |
| Qwen3-14B Q6_K | 33727 | 5608.78 | 5191.51 |

Perplexity, APA e71624b, eps 0.01, `imp-cli --perplexity`, chunk 2048: Llama-3.2-3B 18.2760 -> 18.2836, Qwen3-8B
10.7522 -> 10.7549 (imp `tools/analysis/ppl_corpus_45k.txt`, 44994 bytes, plain); Qwen3-14B long8_32k 2.5249 ->
2.5248.

APA 0.2.0 in imp, Llama-3.2-3B Q8_0, 106451-token prompt, `--max-seq-len 108000`, prefill ms:

| Mode | FA2 | APA eps 0.005 |
|---|---|---|
| default (sparse prefill) | 6661.95 | 5241.18 |
| dense (`sparse_prefill_topk_tokens=0`) | 11659.35 | 6433.23 |

Perplexity, APA 0.2.0 (eps 0.002 / 0.01 rows: 0.1.0, which 0.2.0 repeats exactly at eps 0.005), corpus
`ppl_corpus_45k_gemma4_turn.txt` (46359 bytes, chat-turn wrapped; not the imp file above), 13-14k tokens:

| Variant | Llama-3.2-3B Q8_0 | Qwen3-8B Q8_0 | default-mode prefill ms |
|---|---|---|---|
| FA2 (f16 accumulate) | 17.3644 | 10.7974 | 6661.95 |
| FA2 (f32 accumulate) | 17.3732 | 10.8110 | |
| APA, every tile exact | 17.3741 | 10.7868 | |
| APA eps 0.005 | 17.3423 | 10.7869 | 5241.18 |
| APA eps 0.002 | 17.3556 | 10.7855 | 5915.15 |
| APA eps 0.01 | 17.3385 | 10.7728 | 4863.56 |

## API (`include/apa/apa.cuh`, namespace `apa`)

| Call | Use |
|---|---|
| `prefill(Q, K, V, O, p, eps, ws, ws_bytes, stream)` | flat K/V; `ws` holds `workspace_bytes(p)` |
| `prefill_paged(Q, k_pool, v_pool, block_table, block_size, k_tail, v_tail, tail, O, p, eps, ws, ws_bytes, stream)` | keys `[0, tail)` from a paged FP16 pool, `[tail, Skv)` flat |
| `prefill_incremental(Q, reader, O, p, eps, st, ws, ws_bytes, stream)` | chunked prefill with a per-layer tile cache (`KvState`, `kv_state_bytes` / `kv_state_carve`); a chunk quantizes only its new keys; K mean and head scales rebuilt when the context grows by `APA_KV_RESTAT` (1.125); lc_122880_2, eps 0.005, all pairs, last chunk: min cos 0.985519, 0 pairs < 0.9 (full requantization per chunk: 0.986760, 0; 0.4.0: 0.821417, 24; 0.3.0: 0.679235, 275); restat chunks bitwise equal to `prefill` |
| `supported(p, kv)` | shape gate; every entry point returns false when it declines |

`eps`: tile share of the running row sum above which a tile is exact; use 0.005 (table above).

## Build and bench

| Task | Command |
|---|---|
| Build `build/bench_apa` | `sh bench/build.sh` (needs an nvcc >= 12.9 image, `IMAGE=...`) |
| Run on a dump | `docker run --rm --gpus all -v "$PWD":/w -w /w <image> ./build/bench_apa dump.bin` |
| Test matrix (all dumps: determinism, batch 2, paged, chunked, overflow redo, accuracy) | `DUMPS=dir REF=bin sh bench/test.sh` (REF: earlier build, same kernel) |
| One eps | `APA_EPS=0.005` |
| Short context | `APA_KV=8192` (first N keys) |
| Paged path, bitwise vs flat | `APA_PAGED=16` (block size; `APA_NOSHUF=1`: blocks in order) |
| Chunked prefill + tile cache | `APA_CHUNKS=1`; `=2` also compares every chunk with a full requantization (bitwise); `=3` also scores every chunk vs FP32 |
| Batch 2 (batch 1 = V negated), bitwise vs B = 1 | `APA_B2=1` |
| imp sparse prefill call: past cut to B tokens of 16-key pages (sink, recent, top by q.k) | `APA_SPARSE=B` |
| Raw O per eps to a file / pass-2 union per warp group | `APA_OUT=file` / `APA_UNION=1` |
| Prep: exact stats (0.3.0) / forced overflow redo (test) | `NVFLAGS=-DAPA_PREP_SAMPLE=0` / `-DAPA_PREP_HEADROOM=0.015625f` |
| Error source of the top-20 pairs (host emulation per quantized operand) | `APA_FULL=1 APA_ERRSRC=1` |
| K mean over the first N keys / keys [N, Skv) (diagnostic) | `NVFLAGS=-DAPA_DBG_KMEAN_LEN=N` / `-DAPA_DBG_KMEAN_FROM=N` |
| Determinism (5 reruns, bitwise) | `APA_DET=1` |
| Worst sampled row: norms, V cancellation, hot/cold mass | `APA_DIAG=1` |
| Every (row, head) pair vs FP32: histogram, relL2, top-20; with `APA_PAGED` / `APA_CHUNKS` also those paths | `APA_FULL=1` (`APA_REF_CACHE=file` caches the reference) |
| Pass-1 row export (m, lambda, l_cold) in `APA_FULL` top-20 | `NVFLAGS=-DAPA_DBG sh bench/build.sh` |
| Pass-2 O in fp32 across tiles (exactness studies) | `NVFLAGS=-DAPA_P2_F32O=1 sh bench/build.sh` |
| APA 0.2.0 cold frame (A/B) | `NVFLAGS=-DAPA_HOT_DROP=0 sh bench/build.sh` |

Dump format: `int32 n, kv, nh, nkv, hd, off` then FP16 `Q [n][nh][hd]`, `K [kv][nkv][hd]`, `V [kv][nkv][hd]`.

## License and attribution

[MPL-2.0](LICENSE): file-level copyleft. Modified APA files stay MPL-2.0 and keep their license header; APA
can be combined with code under other licenses. Credit as given in [NOTICE](NOTICE); citation metadata in
[CITATION.cff](CITATION.cff). Paper: [paper/apa.pdf](paper/apa.pdf) (source `paper/apa.tex`, build `sh paper/build.sh`).
