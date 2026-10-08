// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the
// MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Copyright (c) 2026 Raphael Friedmann (github.com/kekzl). APA: https://github.com/kekzl/apa
// bench_apa dump.bin: APA on real Q/K/V (dump: imp lc_*.bin, causal) at eps exact / 1e-3 / 3e-3 / 1e-2.
// Accuracy vs FP32 on 512 sampled rows, hot share of active warp-tiles, prep / attention time (min of 5x5).
#include "../include/apa/apa.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

using T = __half;

__global__ void ref_kernel(const T* Q, const T* K, const T* V, int Sq, int Skv, int H, int Hkv, int off,
                           float scale, const int2* rows, float* scratch, float* out) {
  constexpr int D = 128;
  __shared__ float q[D];
  __shared__ float red[256];
  const int2 rw = rows[blockIdx.x];  // (s, h)
  const int tid = threadIdx.x, hk = rw.y / (H / Hkv);
  float* sc = scratch + (size_t)blockIdx.x * Skv;
  if (tid < D) q[tid] = __half2float(Q[((size_t)rw.x * H + rw.y) * D + tid]);
  __syncthreads();
  const int n = min(off + rw.x + 1, Skv);
  float mx = -INFINITY;
  for (int j = tid; j < n; j += 256) {
    float a = 0;
    for (int d = 0; d < D; ++d) a += q[d] * __half2float(K[((size_t)j * Hkv + hk) * D + d]);
    sc[j] = a * scale;
    mx = fmaxf(mx, sc[j]);
  }
  red[tid] = mx;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  float l = 0;
  for (int j = tid; j < n; j += 256) l += (sc[j] = expf(sc[j] - mx));
  red[tid] = l;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (tid < o) red[tid] += red[tid + o];
    __syncthreads();
  }
  l = red[0];
  if (tid < D) {
    float a = 0;
    for (int j = 0; j < n; ++j) a += sc[j] * __half2float(V[((size_t)j * Hkv + hk) * D + tid]);
    out[(size_t)blockIdx.x * D + tid] = a / l;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: bench_apa dump.bin\n");
    return 1;
  }
  FILE* f = std::fopen(argv[1], "rb");
  int hdr[6];
  if (!f || std::fread(hdr, 4, 6, f) != 6) return 1;
  const int n = hdr[0], nh = hdr[2], nkv = hdr[3], hd = hdr[4];
  int kv = hdr[1], off = hdr[5];
  if (hd != 128) return 1;
  const size_t nq = (size_t)n * nh * hd, nk = (size_t)kv * nkv * hd;
  std::vector<T> hq(nq), hk(nk), hv(nk);
  if (std::fread(hq.data(), 2, nq, f) != nq || std::fread(hk.data(), 2, nk, f) != nk ||
      std::fread(hv.data(), 2, nk, f) != nk)
    return 1;
  std::fclose(f);
  if (const char* e = std::getenv("APA_KV")) {  // keep the first N keys, Q rows at N - n .. N - 1 (fixed-cost study)
    kv = std::max(n, std::min(kv, std::atoi(e)));
    off = kv - n;
  }
  T *Q, *K, *V, *O;
  CK(cudaMalloc(&Q, nq * 2));
  CK(cudaMalloc(&K, nk * 2));
  CK(cudaMalloc(&V, nk * 2));
  CK(cudaMalloc(&O, nq * 2));
  CK(cudaMemcpy(Q, hq.data(), nq * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(K, hk.data(), nk * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(V, hv.data(), nk * 2, cudaMemcpyHostToDevice));
  const float scale = 1.f / std::sqrt(128.f);

  // reference rows
  const int ns = 512;
  std::mt19937 rng(7);
  std::vector<int2> rows(ns);
  for (auto& r : rows) r = make_int2(rng() % n, rng() % nh);
  int2* drows;
  float *scr, *ref;
  CK(cudaMalloc(&drows, ns * sizeof(int2)));
  CK(cudaMalloc(&scr, (size_t)ns * kv * 4));
  CK(cudaMalloc(&ref, (size_t)ns * hd * 4));
  CK(cudaMemcpy(drows, rows.data(), ns * sizeof(int2), cudaMemcpyHostToDevice));
  ref_kernel<<<ns, 256>>>(Q, K, V, n, kv, nh, nkv, off, scale, drows, scr, ref);
  CK(cudaDeviceSynchronize());
  std::vector<float> hr((size_t)ns * hd);
  CK(cudaMemcpy(hr.data(), ref, hr.size() * 4, cudaMemcpyDeviceToHost));
  auto accuracy = [&](double& cmin) {
    std::vector<T> ho(nq);
    CK(cudaMemcpy(ho.data(), O, nq * 2, cudaMemcpyDeviceToHost));
    double dot = 0, na = 0, nb = 0;
    cmin = 1;
    for (int i = 0; i < ns; ++i) {
      double d = 0, x = 0, y = 0;
      for (int e = 0; e < hd; ++e) {
        const double a = hr[(size_t)i * hd + e], b = __half2float(ho[((size_t)rows[i].x * nh + rows[i].y) * hd + e]);
        d += a * b, x += a * a, y += b * b;
      }
      cmin = std::min(cmin, d / std::sqrt(x * y));
      dot += d, na += x, nb += y;
    }
    return dot / std::sqrt(na * nb);
  };
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  auto time = [&](auto fn) {
    fn();
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
      CK(cudaEventRecord(e0));
      for (int i = 0; i < 5; ++i) fn();
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      best = std::min(best, ms / 5);
    }
    return best;
  };
  double pairs = 0;
  for (int i = 0; i < n; ++i) pairs += std::min(off + i + 1, kv);
  const double flops = 4.0 * nh * pairs * hd;
  std::printf("dump n %d kv %d off %d nh %d nkv %d\n", n, kv, off, nh, nkv);

  // ---- APA
  apa::Problem p{1, n, kv, nh, nkv, 128, off, true, scale};
  apa::KvSource akv{};
  akv.k = K;
  akv.v = V;
  const size_t wsb = apa::workspace_bytes(p);
  void* ws;
  CK(cudaMalloc(&ws, wsb));
  const apa::Workspace w = apa::carve(p, ws);
  const float tprep = time([&] { CK(apa::prep(Q, akv, p, w, 0)); });
  std::vector<float> epss{-1.f, 1e-3f, 3e-3f, 1e-2f};
  if (const char* e = std::getenv("APA_EPS")) epss = {(float)std::atof(e)};  // one eps (profiling)
  for (float eps : epss) {
    CK(apa::prep(Q, akv, p, w, 0));
    CK(apa::attn(Q, K, V, O, p, w, eps, 0));
    CK(cudaDeviceSynchronize());
    double cmin;
    const double c = accuracy(cmin);
    // hot share: set bits over active (warp, tile) pairs
    const size_t nw = (size_t)nkv * w.nqb * 12, words = nw * w.W;
    std::vector<uint32_t> hm(words);
    CK(cudaMemcpy(hm.data(), w.warp_hot, words * 4, cudaMemcpyDeviceToHost));
    double hot = 0, act = 0;
    const int G = nh / nkv, R = n * G;
    for (size_t wi = 0; wi < nw; ++wi) {
      const int qb = (int)((wi / 12) % w.nqb), wp = (int)(wi % 12), row0 = qb * 192 + wp * 16;
      if (row0 >= R) continue;
      const int pmax = off + std::min(row0 + 15, R - 1) / G;
      const int nt = std::min(pmax / 64 + 1, (kv + 63) / 64);
      act += nt;
      for (int t = 0; t < nt; ++t) hot += (hm[wi * w.W + t / 32] >> (t % 32)) & 1u;
    }
    // pass-2 CTA union share: tiles hot for any of 4 consecutive warps (APA_P2W) over active pairs
    double chot = 0, cact = 0;
    for (size_t ci = 0; ci < nw / 4; ++ci) {
      const int qb = (int)((ci / 3) % w.nqb), row0 = qb * 192 + (int)(ci % 3) * 64;
      if (row0 >= R) continue;
      const int pmax = off + (std::min(row0 + 64, R) - 1) / G, nt = std::min(pmax / 64 + 1, (kv + 63) / 64);
      cact += nt;
      for (int t = 0; t < nt; ++t) {
        uint32_t u = 0;
        for (int i = 0; i < 4; ++i) u |= hm[(ci * 4 + i) * w.W + t / 32];
        chot += (u >> (t % 32)) & 1u;
      }
    }
    const float ta = time([&] { CK(apa::attn(Q, K, V, O, p, w, eps, 0)); });
    const float t2 = time([&] { CK(apa::pass2(Q, K, V, O, p, w, eps, 0)); });
    std::printf("apa eps %8.0e: cos %.6f min %.6f  hot %5.1f %% p2cta %5.1f %%  attn %.3f ms (pass2 %.3f) %.1f TOPS"
                "  (prep %.3f ms)\n",
                eps, c, cmin, 100 * hot / act, 100 * chot / cact, ta, t2, flops / ta * 1e-9, tprep);
  }
  // APA_DET=1: prep + attention 5x at the last eps, outputs compared bitwise with the first run.
  if (std::getenv("APA_DET")) {
    std::vector<T> h0(nq), h1(nq);
    size_t diff = 0;
    for (int run = 0; run < 5; ++run) {
      CK(apa::prep(Q, akv, p, w, 0));
      CK(apa::attn(Q, K, V, O, p, w, epss.back(), 0));
      CK(cudaMemcpy(run ? h1.data() : h0.data(), O, nq * 2, cudaMemcpyDeviceToHost));
      if (run)
        for (size_t i = 0; i < nq; ++i) diff += __half_as_ushort(h0[i]) != __half_as_ushort(h1[i]);
    }
    std::printf("determinism: %zu of %zu output elements differ over 4 reruns\n", diff, 4 * nq);
  }
  // APA_PAGED=bs: keys [0, off) from a paged pool (blocks shuffled), [off, kv) flat as the tail (imp's chunk);
  // prefill_paged vs flat prefill (prep + attn) at the last eps.
  if (const char* e = std::getenv("APA_PAGED")) {
    const int bs = std::atoi(e), nblk = (off + bs - 1) / bs;
    const size_t row = (size_t)nkv * hd;
    std::vector<int> bt(nblk);
    for (int i = 0; i < nblk; ++i) bt[i] = i;
    if (!std::getenv("APA_NOSHUF")) std::shuffle(bt.begin(), bt.end(), rng);
    T *kp, *vp;
    int* dbt;
    CK(cudaMalloc(&kp, (size_t)nblk * bs * row * 2));
    CK(cudaMalloc(&vp, (size_t)nblk * bs * row * 2));
    CK(cudaMalloc(&dbt, nblk * 4));
    CK(cudaMemcpy(dbt, bt.data(), nblk * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < nblk; ++i) {
      const int n_rows = std::min(bs, off - i * bs);
      CK(cudaMemcpy(kp + (size_t)bt[i] * bs * row, K + (size_t)i * bs * row, n_rows * row * 2, cudaMemcpyDeviceToDevice));
      CK(cudaMemcpy(vp + (size_t)bt[i] * bs * row, V + (size_t)i * bs * row, n_rows * row * 2, cudaMemcpyDeviceToDevice));
    }
    const float eps = epss.back();
    const float tf = time([&] {
      if (!apa::prefill(Q, K, V, O, p, eps, ws, wsb, 0)) std::exit(2);
    });
    double cmin;
    const double cf = accuracy(cmin);
    const float tp = time([&] {
      if (!apa::prefill_paged(Q, kp, vp, dbt, bs, K + (size_t)off * row, V + (size_t)off * row, off, O, p, eps,
                              ws, wsb, 0))
        std::exit(3);
    });
    CK(cudaDeviceSynchronize());
    double cmin2;
    const double cp = accuracy(cmin2);
    std::printf("paged bs %d eps %.0e: flat prep+attn %.3f ms cos %.6f min %.6f | paged %.3f ms cos %.6f min %.6f\n",
                bs, eps, tf, cf, cmin, tp, cp, cmin2);
    CK(cudaFree(kp));
    CK(cudaFree(vp));
    CK(cudaFree(dbt));
  }
  // APA_CHUNKS=1: chunked prefill over the dump (chunks of n queries, the dump's Q rows each time, keys
  // [0, (c+1) n)); full requantization per chunk vs prefill_incremental (KvState). Accuracy = last chunk.
  if (std::getenv("APA_CHUNKS") && kv % n == 0) {
    const float eps = epss.back();
    const int nc = kv / n;
    auto chunk = [&](int c) {
      apa::Problem pc = p;
      pc.Skv = (c + 1) * n;
      pc.q_offset = pc.Skv - n;
      return pc;
    };
    const float tf = time([&] {
      for (int c = 0; c < nc; ++c)
        if (!apa::prefill(Q, K, V, O, chunk(c), eps, ws, wsb, 0)) std::exit(4);
    });
    double cmin;
    const double cf = accuracy(cmin);
    void* sb;
    CK(cudaMalloc(&sb, apa::kv_state_bytes(1, nkv, 128, kv)));
    apa::KvState st = apa::kv_state_carve(sb, 1, nkv, 128, kv);
    const apa::FlatKV<T> rd{K, V};
    const float ti = time([&] {
      st.len = st.stats_len = 0;
      for (int c = 0; c < nc; ++c)
        if (!apa::prefill_incremental(Q, rd, O, chunk(c), eps, st, ws, wsb, 0)) std::exit(5);
    });
    CK(cudaDeviceSynchronize());
    double cmin2;
    const double ci = accuracy(cmin2);
    std::printf("chunks %d x %d eps %.0e: full requant %.2f ms cos %.6f min %.6f | incremental %.2f ms cos %.6f min %.6f\n",
                nc, n, eps, tf, cf, cmin, ti, ci, cmin2);
    // stale cache: keys shifted by one token, q_offset == st.len: the fingerprint must force a full redo
    apa::Problem ps = chunk(nc - 2);
    const apa::FlatKV<T> rs{K + (size_t)nkv * 128, V + (size_t)nkv * 128};
    st.len = st.stats_len = ps.q_offset;
    if (!apa::prefill_incremental(Q, rs, O, ps, eps, st, ws, wsb, 0)) std::exit(6);
    std::vector<T> h1(nq), h2(nq);
    CK(cudaMemcpy(h1.data(), O, nq * 2, cudaMemcpyDeviceToHost));
    if (!apa::prefill(Q, rs.k, rs.v, O, ps, eps, ws, wsb, 0)) std::exit(7);
    CK(cudaMemcpy(h2.data(), O, nq * 2, cudaMemcpyDeviceToHost));
    double d = 0, a = 0, bb = 0;
    for (size_t i = 0; i < nq; ++i) {
      const double x = __half2float(h1[i]), y = __half2float(h2[i]);
      d += x * y, a += x * x, bb += y * y;
    }
    std::printf("stale-cache redo vs fresh prefill: cos %.6f\n", d / std::sqrt(a * bb));
    CK(cudaFree(sb));
  }
  CK(cudaFree(ws));

  return 0;
}
