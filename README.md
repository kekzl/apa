# APA: adaptive-precision attention

Causal flash-attention prefill for RTX 5090 (sm_120a), header-only CUDA. Long-context attention mass sits in
few KV tiles; APA computes those exactly and the rest in FP4.

| Step | What |
|---|---|
| Pass 1 | all-FP4 flash attention (`mma.sync kind::mxf4nvf4`) over all 64-key tiles; a tile whose share of the running row sum exceeds `eps` is hot: no P·V here, bit in a per-warp mask |
| Pass 2 | exact FP16 attention (`mma.sync m16n8k16`) over the hot tiles only |
| Merge | log-sum-exp of the cold-tile partials (pass 1) and the hot result (pass 2) |
| Launch | one fused kernel: pass-1 CTAs, then persistent pass-2 workers on ticket counters and per-q-block ready flags |

Scope: head dim 128, causal (`q_offset` = position of Q row 0), GQA (`H % Hkv == 0`), FP16 Q/K/V. K/V flat, or a
paged FP16 pool through a block table plus a flat tail (chunked prefill). Pass-1 tile order: sink tile first,
then the diagonal backwards.

## Results

Standalone, Llama-3.2-3B attention dumps (2048 queries at kv 122880, 24 / 8 heads), eps 0.005, cos vs FP32:

| Dump | cos mean | cos min | hot tiles | attn ms |
|---|---|---|---|---|
| lc_122880_0 | 0.999000 | 0.898838 | 10.4 % | 4.467 |
| lc_122880_1 | 0.999626 | 0.970893 | 9.7 % | 4.583 |
| lc_122880_2 | 0.997437 | -0.444615 | 6.0 % | 4.760 |

Same code with every tile exact (`eps < 0`): lc_122880_1 10.725 ms, cos 0.999634.

In [imp](https://github.com/kekzl/imp) (`attention.apa_eps`), Llama-3.2-3B Q8_0, 106451-token prompt,
`--max-seq-len 108000`, prefill ms:

| Mode | FA2 | APA eps 0.005 |
|---|---|---|
| default (sparse prefill) | 6661.95 | 5300.33 |
| dense (`sparse_prefill_topk_tokens=0`) | 11659.35 | 6474.33 |

Perplexity (imp, ppl_corpus_45k, 13-14k tokens):

| Variant | Llama-3.2-3B Q8_0 | Qwen3-8B Q8_0 | default-mode prefill ms |
|---|---|---|---|
| FA2 (f16 accumulate) | 17.3644 | 10.7974 | 6661.95 |
| FA2 (f32 accumulate) | 17.3732 | 10.8110 | |
| APA, every tile exact | 17.3741 | 10.7868 | |
| APA eps 0.005 | 17.3432 | 10.8034 | 5300.33 |
| APA eps 0.002 | 17.3556 | 10.7855 | 5915.15 |
| APA eps 0.01 | 17.3385 | 10.7728 | 4863.56 |

## API (`include/apa/apa.cuh`, namespace `apa`)

| Call | Use |
|---|---|
| `prefill(Q, K, V, O, p, eps, ws, ws_bytes, stream)` | flat K/V; `ws` holds `workspace_bytes(p)` |
| `prefill_paged(Q, k_pool, v_pool, block_table, block_size, k_tail, v_tail, tail, O, p, eps, ws, ws_bytes, stream)` | keys `[0, tail)` from a paged FP16 pool, `[tail, Skv)` flat |
| `prefill_incremental(Q, reader, O, p, eps, st, ws, ws_bytes, stream)` | chunked prefill with a per-layer tile cache (`KvState`, `kv_state_bytes` / `kv_state_carve`); a chunk quantizes only its new keys |
| `supported(p, kv)` | shape gate; every entry point returns false when it declines |

`eps`: tile share of the running row sum above which a tile is exact; 0.005 is the measured trade-off above.

## Build and bench

| Task | Command |
|---|---|
| Build `build/bench_apa` | `sh bench/build.sh` (needs an nvcc >= 12.9 image, `IMAGE=...`) |
| Run on a dump | `docker run --rm --gpus all -v "$PWD":/w -w /w <image> ./build/bench_apa dump.bin` |
| One eps | `APA_EPS=0.005` |
| Short context | `APA_KV=8192` (first N keys) |
| Paged path | `APA_PAGED=16` (block size) |
| Chunked prefill + tile cache | `APA_CHUNKS=1` |

Dump format: `int32 n, kv, nh, nkv, hd, off` then FP16 `Q [n][nh][hd]`, `K [kv][nkv][hd]`, `V [kv][nkv][hd]`.

## License and attribution

[MPL-2.0](LICENSE): file-level copyleft. Modified APA files stay MPL-2.0 and keep their license header; APA
can be combined with code under other licenses. Credit as given in [NOTICE](NOTICE); citation metadata in
[CITATION.cff](CITATION.cff).
