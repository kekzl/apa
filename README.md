# APA: adaptive-precision attention

Causal flash-attention prefill for RTX 5090 (sm_120a), header-only CUDA. Long-context attention mass sits in
few KV tiles; APA computes those exactly and the rest in FP4.

| Step | What |
|---|---|
| Pass 1 | all-FP4 flash attention (`mma.sync kind::mxf4nvf4`) over all 64-key tiles, Q as two E2M1 terms on its 64 largest channels (`APA_Q2`); a tile whose share of the running row sum exceeds `eps` is hot: no P·V here, bit in a per-warp mask |
| Pass 2 | exact FP16 attention (`mma.sync m16n8k16`) over the hot tiles only; one K/V tile stream per q block shared by its 12 warps |
| Merge | log-sum-exp of the cold-tile partials (pass 1) and the hot result (pass 2) |
| Launch | one fused kernel: pass-1 CTAs, then persistent pass-2 workers on ticket counters and per-q-block ready flags |

Scope: head dim 128, causal (`q_offset` = position of Q row 0), GQA (`H % Hkv == 0`), FP16 Q/K/V. K/V flat, or a
paged FP16 pool through a block table plus a flat tail (chunked prefill). Pass-1 tile order: sink tile first,
then the diagonal backwards.

## Results

Standalone, Llama-3.2-3B attention dumps (2048 queries at kv 122880, 65536 or 32768, 24 / 8 heads), eps 0.005, every
(row, head) pair (49152) vs FP32 (`APA_FULL=1`), one run, `APA_Q2=2` (bench/test.sh run; PERF_LOG.md, "Phase 6"):

| Dump | pooled cos | mean cos | min cos | pairs < 0.9 | hot tiles | attn ms | prep ms |
|---|---|---|---|---|---|---|---|
| lc_122880_0 | 0.999645 | 0.998743 | 0.963487 | 0 | 10.3 % | 4.969 | 0.520 |
| lc_122880_1 | 0.999961 | 0.999935 | 0.996712 | 0 | 9.3 % | 5.193 | 0.519 |
| lc_122880_2 | 0.999875 | 0.999577 | 0.977037 | 0 | 5.9 % | 5.202 | 0.521 |
| lc_65536_0 | 0.999729 | 0.999127 | 0.977662 | 0 | 19.3 % | 3.115 | 0.318 |
| lc_65536_1 | 0.999969 | 0.999947 | 0.997799 | 0 | 13.4 % | 3.144 | 0.315 |
| lc_65536_2 | 0.999893 | 0.999648 | 0.988146 | 0 | 8.9 % | 3.167 | 0.320 |
| lc_32768_0 | 0.999852 | 0.999505 | 0.984118 | 0 | 39.3 % | 2.524 | 0.196 |
| lc_32768_1 | 0.999978 | 0.999956 | 0.997834 | 0 | 23.5 % | 1.921 | 0.197 |
| lc_32768_2 | 0.999924 | 0.999708 | 0.984406 | 0 | 17.3 % | 1.928 | 0.196 |

Single-term Q (`APA_Q2=0`, = 0.4.0 bitwise), same dumps: 137 pairs < 0.9 (lc_122880_2: 13, lc_65536_0: 124), min
cos 0.773144 (lc_65536_0); two-term attn +4.8 to +15.2 % (paper, "Error source of the worst rows").

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

In [imp](https://github.com/kekzl/imp) (`attention.apa_eps`), measured by the imp integration (relayed, PERF_LOG.md).
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
| `prefill_incremental(Q, reader, O, p, eps, st, ws, ws_bytes, stream)` | chunked prefill with a per-layer tile cache (`KvState`, `kv_state_bytes` / `kv_state_carve`); a chunk quantizes only its new keys; K mean and head scales rebuilt when the context grows by `APA_KV_RESTAT` (1.125); lc_122880_2, eps 0.005, all pairs, last chunk: min cos 0.981605, 0 pairs < 0.9 (full requantization per chunk: 0.977037, 0; single-term Q: 0.821417, 24; 0.3.0: 0.679235, 275); restat chunks bitwise equal to `prefill` |
| `supported(p, kv)` | shape gate; every entry point returns false when it declines |

`eps`: tile share of the running row sum above which a tile is exact; 0.005 is the measured trade-off above.
0.01 with `APA_Q2=2`: up to 65536 keys faster than single-term Q at 0.005 (lc_32768_0 1.670 vs 2.364 ms), no pair
< 0.9 on 12 dumps.

## Build and bench

| Task | Command |
|---|---|
| Build `build/bench_apa` | `sh bench/build.sh` (needs an nvcc >= 12.9 image, `IMAGE=...`) |
| Run on a dump | `docker run --rm --gpus all -v "$PWD":/w -w /w <image> ./build/bench_apa dump.bin` |
| Test matrix (all dumps: determinism, batch 2, paged, chunked, overflow redo, accuracy) | `DUMPS=dir REF=old_bin sh bench/test.sh` |
| One eps | `APA_EPS=0.005` |
| Short context | `APA_KV=8192` (first N keys) |
| Paged path, bitwise vs flat | `APA_PAGED=16` (block size; `APA_NOSHUF=1`: blocks in order) |
| Chunked prefill + tile cache | `APA_CHUNKS=1`; `=2` also compares every chunk with a full requantization (bitwise) |
| Batch 2 (batch 1 = V negated), bitwise vs B = 1 | `APA_B2=1` |
| Raw O per eps to a file / pass-2 union per warp group | `APA_OUT=file` / `APA_UNION=1` |
| Prep: exact stats (0.3.0) / forced overflow redo (test) | `NVFLAGS=-DAPA_PREP_SAMPLE=0` / `-DAPA_PREP_HEADROOM=0.015625f` |
| Q terms: single (0.4.0) / two on all channels / two on 64 channels (default) | `NVFLAGS=-DAPA_Q2=0` / `=1` / `=2` |
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
