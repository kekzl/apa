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
#include <string>
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
  int worst = 0;  // sample index of the last cmin (APA_DIAG)
  std::vector<float> rall;  // APA_FULL: FP32 reference of every (row, head) pair
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
      if (d / std::sqrt(x * y) < cmin) cmin = d / std::sqrt(x * y), worst = i;
      dot += d, na += x, nb += y;
    }
    return dot / std::sqrt(na * nb);
  };
  // APA_FULL summary of the current O over all pairs (needs rall from the eps loop); for the paged / chunk paths
  auto full_summary = [&](const char* tag) {
    if (rall.empty()) return;
    std::vector<T> ho(nq);
    CK(cudaMemcpy(ho.data(), O, nq * 2, cudaMemcpyDeviceToHost));
    const size_t np = (size_t)n * nh;
    double dot = 0, na = 0, nb = 0, csum = 0, cm = 1;
    size_t b9 = 0, b0 = 0;
    for (size_t i = 0; i < np; ++i) {
      double d = 0, x = 0, y = 0;
      for (int e = 0; e < hd; ++e) {
        const double a = rall[i * hd + e], b = __half2float(ho[i * hd + e]);
        d += a * b, x += a * a, y += b * b;
      }
      const double c = d / std::sqrt(x * y);
      dot += d, na += x, nb += y, csum += c, cm = std::min(cm, c), b9 += c < 0.9, b0 += c < 0;
    }
    std::printf("  full %s: pooled cos %.6f mean cos %.6f min %.6f cos<0.9 %zu cos<0 %zu\n", tag,
                dot / std::sqrt(na * nb), csum / np, cm, b9, b0);
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
  apa::Workspace w = apa::carve(p, ws);
#ifdef APA_DBG
  CK(cudaMalloc(&w.dbg, (size_t)n * nh * sizeof(float4)));  // [nkv][R] rows, R = n * G
#endif
  const float tprep = time([&] { CK(apa::prep(Q, akv, p, w, 0)); });
  int povf = 0;
  CK(cudaMemcpy(&povf, w.ovf, 4, cudaMemcpyDeviceToHost));
  std::printf("prep: sampled-stats overflow redo %d\n", povf);
  std::vector<float> epss{-1.f, 1e-3f, 3e-3f, 1e-2f};
  if (const char* e = std::getenv("APA_EPS")) epss = {(float)std::atof(e)};  // one eps (profiling)
  for (float eps : epss) {
    CK(apa::prep(Q, akv, p, w, 0));
    CK(apa::attn(Q, K, V, O, p, w, eps, 0));
    CK(cudaDeviceSynchronize());
    double cmin;
    const double c = accuracy(cmin);
    if (const char* fo = std::getenv("APA_OUT")) {  // raw O of this eps (bitwise A/B of kernel variants)
      std::vector<T> ho(nq);
      CK(cudaMemcpy(ho.data(), O, nq * 2, cudaMemcpyDeviceToHost));
      if (FILE* g = std::fopen((std::string(fo) + "_" + std::to_string(eps)).c_str(), "wb")) {
        std::fwrite(ho.data(), 2, nq, g);
        std::fclose(g);
      }
    }
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
    // pass-2 load share: tiles hot for any of the 12 warps of a q block (pass2_cta stream) over active pairs
    double chot = 0, cact = 0;
    for (size_t ci = 0; ci < nw / 12; ++ci) {
      const int qb = (int)(ci % w.nqb), row0 = qb * 192;
      if (row0 >= R) continue;
      const int pmax = off + (std::min(row0 + 192, R) - 1) / G, nt = std::min(pmax / 64 + 1, (kv + 63) / 64);
      cact += nt;
      for (int t = 0; t < nt; ++t) {
        uint32_t u = 0;
        for (int i = 0; i < 12; ++i) u |= hm[(ci * 12 + i) * w.W + t / 32];
        chot += (u >> (t % 32)) & 1u;
      }
    }
    // APA_UNION=1: union share over groups of gs consecutive warps of a CTA (tiles loaded per group / active pairs)
    if (std::getenv("APA_UNION")) {
      std::printf("union eps %8.0e:", eps);
      for (int gs : {1, 2, 3, 4, 6, 12}) {
        double uh = 0, ua = 0;
        for (size_t ci = 0; ci < nw / gs; ++ci) {
          const int qb = (int)((ci * gs / 12) % w.nqb), row0 = qb * 192 + (int)(ci * gs % 12) * 16;
          if (row0 >= R) continue;
          const int pmax = off + (std::min(row0 + 16 * gs, R) - 1) / G, nt = std::min(pmax / 64 + 1, (kv + 63) / 64);
          ua += nt;
          for (int t = 0; t < nt; ++t) {
            uint32_t u = 0;
            for (int i = 0; i < gs; ++i) u |= hm[(ci * gs + i) * w.W + t / 32];
            uh += (u >> (t % 32)) & 1u;
          }
        }
        std::printf("  %d: %5.1f %%", gs, 100 * uh / ua);
      }
      // per q block: own tiles of the busiest warp and of the busiest SMSP (warps w, w+4, w+8) vs the 12-warp union
      double su = 0, sa = 0, sm = 0, ss = 0;
      for (size_t qi = 0; qi < nw / 12; ++qi) {
        int own[12] = {0}, un = 0;
        for (int t = 0; t < (kv + 63) / 64; ++t) {
          uint32_t u = 0;
          for (int i = 0; i < 12; ++i) {
            const uint32_t bit = (hm[(qi * 12 + i) * w.W + t / 32] >> (t % 32)) & 1u;
            own[i] += bit, u |= bit;
          }
          un += u;
        }
        int mw = 0, ms = 0, tot = 0;
        for (int i = 0; i < 12; ++i) mw = std::max(mw, own[i]), tot += own[i];
        for (int s = 0; s < 4; ++s) ms = std::max(ms, own[s] + own[s + 4] + own[s + 8]);
        su += un, sa += tot / 12.0, sm += mw, ss += ms;
      }
      std::printf("  | per q block: union %.0f, own avg %.0f max %.0f, busiest SMSP %.0f\n", su / (nw / 12),
                  sa / (nw / 12), sm / (nw / 12), ss / (nw / 12));
    }
    const float ta = time([&] { CK(apa::attn(Q, K, V, O, p, w, eps, 0)); });
    const float t2 = time([&] { CK(apa::pass2(Q, K, V, O, p, w, eps, 0)); });
    std::printf("apa eps %8.0e: cos %.6f min %.6f  hot %5.1f %% p2load %5.1f %%  attn %.3f ms (pass2 %.3f) %.1f TOPS"
                "  (prep %.3f ms)\n",
                eps, c, cmin, 100 * hot / act, 100 * chot / cact, ta, t2, flops / ta * 1e-9, tprep);
    // APA_DIAG=1: worst sampled row (min cos): norms, V cancellation, tile mass split by the warp's hot mask.
    if (std::getenv("APA_DIAG")) {
      CK(apa::attn(Q, K, V, O, p, w, eps, 0));
      double cm;
      accuracy(cm);
      const int s = rows[worst].x, h = rows[worst].y, hkv = h / G, nrow = std::min(off + s + 1, kv);
      std::vector<float> sc(nrow);
      CK(cudaMemcpy(sc.data(), scr + (size_t)worst * kv, nrow * 4, cudaMemcpyDeviceToHost));
      std::vector<T> orow(hd);
      CK(cudaMemcpy(orow.data(), O + ((size_t)s * nh + h) * hd, hd * 2, cudaMemcpyDeviceToHost));
      const int r = s * G + h % G, wi = (hkv * w.nqb + r / 192) * 12 + (r % 192) / 16;
      double l = 0;
      for (int j = 0; j < nrow; ++j) l += sc[j];
      std::vector<double> cold(hd, 0.0), hotv(hd, 0.0);
      double pvn = 0, mtop = 0, mhot = 0, mcold_big = 0;
      int nhot = 0, nbig = 0;
      for (int t = 0; t * 64 < nrow; ++t) {
        const bool ht = (hm[wi * w.W + t / 32] >> (t % 32)) & 1u;
        double mt = 0;
        for (int j = t * 64; j < std::min(nrow, t * 64 + 64); ++j) {
          const double pj = sc[j] / l;
          double vn = 0;
          for (int e = 0; e < hd; ++e) {
            const double v = __half2float(hv[((size_t)j * nkv + hkv) * hd + e]);
            vn += v * v;
            (ht ? hotv : cold)[e] += pj * v;
          }
          pvn += pj * std::sqrt(vn);
          mt += pj;
        }
        mtop = std::max(mtop, mt);
        nhot += ht, mhot += ht ? mt : 0;
        if (mt > 0.005) ++nbig, mcold_big += ht ? 0 : mt;
      }
      double rn = 0, on = 0, dn = 0, cn = 0, hn = 0;
      for (int e = 0; e < hd; ++e) {
        const double a = hr[(size_t)worst * hd + e], b = __half2float(orow[e]);
        rn += a * a, on += b * b, dn += (a - b) * (a - b), cn += cold[e] * cold[e], hn += hotv[e] * hotv[e];
      }
      std::printf("  diag s %d h %d cos %.4f |ref| %.4f |out| %.4f relerr %.3f |ref|/sum p|v| %.4f top tile %.3f "
                  "hot %d tiles mass %.3f |hot pv| %.4f |cold pv| %.4f, tiles > 0.005: %d (cold mass %.3f)\n",
                  s, h, cm, std::sqrt(rn), std::sqrt(on), std::sqrt(dn / rn), std::sqrt(rn) / pvn, mtop, nhot,
                  mhot, std::sqrt(hn), std::sqrt(cn), nbig, mcold_big);
    }
    // APA_FULL=1: every (row, head) pair vs the FP32 ref_kernel (batches of 1024, cached in APA_REF_CACHE):
    // cos and relative L2 histogram, top-20 pairs with warp state (pass-1 export in -DAPA_DBG builds), FP32 tile
    // masses, Q / K channel amax spread of the pair's heads.
    if (std::getenv("APA_FULL")) {
      const size_t np = (size_t)n * nh;
      if (rall.empty()) {
        rall.resize(np * hd);
        const char* cache = std::getenv("APA_REF_CACHE");
        FILE* g = cache ? std::fopen(cache, "rb") : nullptr;
        const bool hit = g && std::fread(rall.data(), 4, rall.size(), g) == rall.size();
        if (g) std::fclose(g);
        if (!hit) {
          const int nb = 1024;
          std::vector<int2> pr(nb);
          int2* dpr;
          float *dscr, *dout;
          CK(cudaMalloc(&dpr, nb * sizeof(int2)));
          CK(cudaMalloc(&dscr, (size_t)nb * kv * 4));
          CK(cudaMalloc(&dout, (size_t)nb * hd * 4));
          for (size_t i0 = 0; i0 < np; i0 += nb) {
            const int c = (int)std::min<size_t>(nb, np - i0);
            for (int i = 0; i < c; ++i) pr[i] = make_int2((int)((i0 + i) / nh), (int)((i0 + i) % nh));
            CK(cudaMemcpy(dpr, pr.data(), c * sizeof(int2), cudaMemcpyHostToDevice));
            ref_kernel<<<c, 256>>>(Q, K, V, n, kv, nh, nkv, off, scale, dpr, dscr, dout);
            CK(cudaMemcpy(rall.data() + i0 * hd, dout, (size_t)c * hd * 4, cudaMemcpyDeviceToHost));
          }
          CK(cudaFree(dpr));
          CK(cudaFree(dscr));
          CK(cudaFree(dout));
          if (cache && (g = std::fopen(cache, "wb"))) {
            std::fwrite(rall.data(), 4, rall.size(), g);
            std::fclose(g);
          }
        }
      }
#ifdef APA_DBG
      CK(cudaMemset(w.dbg, 0, (size_t)nkv * R * sizeof(float4)));
#endif
      CK(apa::attn(Q, K, V, O, p, w, eps, 0));
      CK(cudaDeviceSynchronize());
      std::vector<T> ho(nq);
      CK(cudaMemcpy(ho.data(), O, nq * 2, cudaMemcpyDeviceToHost));
      std::vector<std::pair<double, size_t>> cs(np);
      std::vector<double> rel(np);
      double dot = 0, na = 0, nb2 = 0, csum = 0;
      const double edges[] = {0.0, 0.5, 0.9, 0.99, 0.999, 0.9999};
      size_t hist[7] = {0, 0, 0, 0, 0, 0, 0};
      for (size_t i = 0; i < np; ++i) {
        double d = 0, x = 0, y = 0, e2 = 0;
        for (int e = 0; e < hd; ++e) {
          const double a = rall[i * hd + e], b = __half2float(ho[i * hd + e]);
          d += a * b, x += a * a, y += b * b, e2 += (a - b) * (a - b);
        }
        const double c = d / std::sqrt(x * y);
        cs[i] = {c, i};
        rel[i] = std::sqrt(e2 / x);
        dot += d, na += x, nb2 += y, csum += c;
        int k = 0;
        while (k < 6 && c >= edges[k]) ++k;
        ++hist[k];
      }
      std::vector<double> rs = rel;
      std::sort(rs.begin(), rs.end());
      double rsum = 0;
      for (double v : rel) rsum += v;
      std::partial_sort(cs.begin(), cs.begin() + 20, cs.end());
      std::printf("full eps %8.0e: pairs %zu pooled cos %.6f mean cos %.6f min %.6f | cos<0.99 %zu cos<0.9 %zu cos<0 %zu |"
                  " relL2 mean %.5f p50 %.5f p99 %.5f p999 %.5f max %.5f\n",
                  eps, np, dot / std::sqrt(na * nb2), csum / np, cs[0].first, hist[0] + hist[1] + hist[2] + hist[3],
                  hist[0] + hist[1] + hist[2], hist[0], rsum / np, rs[np / 2], rs[np * 99 / 100], rs[np * 999 / 1000],
                  rs[np - 1]);
      std::printf("  hist cos: <0 %zu [0,.5) %zu [.5,.9) %zu [.9,.99) %zu [.99,.999) %zu [.999,.9999) %zu >=.9999 %zu\n",
                  hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6]);
      // per-channel amax spread (max / median over 128 channels) of K per kv head and Q per q head
      auto spread = [&](const std::vector<T>& src, size_t rowsN, int heads, int hsel) {
        std::vector<float> am(hd, 0.f);
        for (size_t r2 = 0; r2 < rowsN; ++r2)
          for (int e = 0; e < hd; ++e)
            am[e] = std::max(am[e], std::fabs(__half2float(src[(r2 * heads + hsel) * hd + e])));
        std::vector<float> s2 = am;
        std::sort(s2.begin(), s2.end());
        return s2[hd - 1] / std::max(s2[hd / 2], 1e-6f);
      };
      std::vector<int2> top(20);
      for (int i = 0; i < 20; ++i) top[i] = make_int2((int)(cs[i].second / nh), (int)(cs[i].second % nh));
      int2* dtop;
      float *dscr, *dout;
      CK(cudaMalloc(&dtop, 20 * sizeof(int2)));
      CK(cudaMalloc(&dscr, (size_t)20 * kv * 4));
      CK(cudaMalloc(&dout, 20 * hd * 4));
      CK(cudaMemcpy(dtop, top.data(), 20 * sizeof(int2), cudaMemcpyHostToDevice));
      ref_kernel<<<20, 256>>>(Q, K, V, n, kv, nh, nkv, off, scale, dtop, dscr, dout);
      std::vector<float> sc20((size_t)20 * kv);
      CK(cudaMemcpy(sc20.data(), dscr, sc20.size() * 4, cudaMemcpyDeviceToHost));
#ifdef APA_DBG
      std::vector<float4> hdbg((size_t)nkv * R);
      CK(cudaMemcpy(hdbg.data(), w.dbg, hdbg.size() * sizeof(float4), cudaMemcpyDeviceToHost));
#endif
      for (int i = 0; i < 20; ++i) {
        const int s = top[i].x, h = top[i].y, hkv = h / G, nrow = std::min(off + s + 1, kv);
        const int r = s * G + h % G, qb = r / 192, wi = (hkv * w.nqb + qb) * 12 + (r % 192) / 16;
        const float* sc = sc20.data() + (size_t)i * kv;
        double l = 0;
        for (int j = 0; j < nrow; ++j) l += sc[j];
        std::vector<double> tm;
        double mcold = 0, tmax = 0, cmax = 0, cpv2 = 0;
        int nhot = 0, ncbig = 0;
        std::vector<double> cpv(hd, 0.0);
        for (int t = 0; t * 64 < nrow; ++t) {
          const bool ht = ((hm[wi * w.W + t / 32] >> (t % 32)) & 1u);
          double mt = 0;
          for (int j = t * 64; j < std::min(nrow, t * 64 + 64); ++j) {
            mt += sc[j] / l;
            if (!ht)
              for (int e = 0; e < hd; ++e) cpv[e] += sc[j] / l * __half2float(hv[((size_t)j * nkv + hkv) * hd + e]);
          }
          tm.push_back(mt);
          tmax = std::max(tmax, mt);
          nhot += ht;
          if (!ht) mcold += mt, cmax = std::max(cmax, mt), ncbig += mt > eps;
        }
        std::sort(tm.rbegin(), tm.rend());
        int n90 = 0;
        for (double acc = 0; n90 < (int)tm.size() && acc < 0.9; ++n90) acc += tm[n90];
        double rn = 0;
        for (int e = 0; e < hd; ++e) rn += (double)rall[cs[i].second * hd + e] * rall[cs[i].second * hd + e], cpv2 += cpv[e] * cpv[e];
        std::printf("  top %2d s %4d h %2d qb %2d warp %d cos %.6f relL2 %.4f | hot %4d/%4zu %s | fp32: cold mass %.4f max "
                    "tile %.4f max cold tile %.4f cold tiles > eps %d n90 %d |cold pv|/|ref| %.3f | spread K %.1f Q %.1f",
                    i, s, h, qb, wi, cs[i].first, rel[cs[i].second], nhot, tm.size(), nhot ? "merge" : "cold-only",
                    mcold, tmax, cmax, ncbig, n90, std::sqrt(cpv2 / rn), spread(hk, kv, nkv, hkv),
                    spread(hq, n, nh, h));
#ifdef APA_DBG
        const float4 dg = hdbg[(size_t)hkv * R + r];
        std::printf(" | fp4: m %.2f lambda %.4g l_cold %.4g cold share %.4f max tile %.4f", dg.x, dg.y, dg.z,
                    dg.y > 0.f ? dg.z / dg.y : 0.f, dg.w);
#endif
        std::printf("\n");
      }
      CK(cudaFree(dtop));
      CK(cudaFree(dscr));
      CK(cudaFree(dout));
    }
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
    full_summary("flat");
    std::vector<T> hfl(nq), hpg(nq);
    CK(cudaMemcpy(hfl.data(), O, nq * 2, cudaMemcpyDeviceToHost));
    const float tp = time([&] {
      if (!apa::prefill_paged(Q, kp, vp, dbt, bs, K + (size_t)off * row, V + (size_t)off * row, off, O, p, eps,
                              ws, wsb, 0))
        std::exit(3);
    });
    CK(cudaDeviceSynchronize());
    double cmin2;
    const double cp = accuracy(cmin2);
    full_summary("paged");
    CK(cudaMemcpy(hpg.data(), O, nq * 2, cudaMemcpyDeviceToHost));
    size_t pdiff = 0;
    for (size_t i = 0; i < nq; ++i) pdiff += __half_as_ushort(hfl[i]) != __half_as_ushort(hpg[i]);
    std::printf("paged vs flat: %zu of %zu output elements differ\n", pdiff, nq);
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
    full_summary("full requant");
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
    full_summary("incremental");
    std::printf("chunks %d x %d eps %.0e: full requant %.2f ms cos %.6f min %.6f | incremental %.2f ms cos %.6f min %.6f\n",
                nc, n, eps, tf, cf, cmin, ti, ci, cmin2);
    // APA_CHUNKS=2: every chunk, incremental vs full requantization (same keys, same Q): min cos over all pairs
    if (std::atoi(std::getenv("APA_CHUNKS")) == 2) {
      std::vector<T> hf(nq), hi(nq);
      double wmin = 1;
      int wc = 0;
      size_t w9 = 0, w99 = 0;
      int neq = 0;  // chunks with bitwise equal output
      st.len = st.stats_len = 0;
      for (int c = 0; c < nc; ++c) {
        if (!apa::prefill(Q, K, V, O, chunk(c), eps, ws, wsb, 0)) std::exit(8);
        CK(cudaMemcpy(hf.data(), O, nq * 2, cudaMemcpyDeviceToHost));
        if (!apa::prefill_incremental(Q, rd, O, chunk(c), eps, st, ws, wsb, 0)) std::exit(9);
        CK(cudaMemcpy(hi.data(), O, nq * 2, cudaMemcpyDeviceToHost));
        double cm = 1;
        size_t b9 = 0, b99 = 0, nd = 0;
        for (size_t i = 0; i < nq; ++i) nd += __half_as_ushort(hf[i]) != __half_as_ushort(hi[i]);
        neq += nd == 0;
        for (size_t i = 0; i < (size_t)n * nh; ++i) {
          double d = 0, x = 0, y = 0;
          for (int e = 0; e < hd; ++e) {
            const double a = __half2float(hf[i * hd + e]), b = __half2float(hi[i * hd + e]);
            d += a * b, x += a * a, y += b * b;
          }
          const double cc = d / std::sqrt(x * y);
          cm = std::min(cm, cc), b9 += cc < 0.9, b99 += cc < 0.99;
        }
        std::printf("  chunk %2d kv %6d stats_len %6d: incr vs requant min cos %.6f cos<0.99 %zu cos<0.9 %zu, %zu differ\n",
                    c, (c + 1) * n, st.stats_len, cm, b99, b9, nd);
        if (cm < wmin) wmin = cm, wc = c;
        w9 = std::max(w9, b9), w99 = std::max(w99, b99);
      }
      std::printf("chunks incr vs requant: worst min cos %.6f (chunk %d), max cos<0.99 %zu, max cos<0.9 %zu, bitwise equal"
                  " %d of %d\n", wmin, wc, w99, w9, neq, nc);
    }
    // stale cache (>= 2 chunks): keys shifted by one token, q_offset == st.len: the fingerprint must force a redo
    if (nc >= 2) {
      apa::Problem ps = chunk(nc - 2);
      const apa::FlatKV<T> rs{K + (size_t)nkv * 128, V + (size_t)nkv * 128};
      st.len = st.stats_len = ps.q_offset;
      if (!apa::prefill_incremental(Q, rs, O, ps, eps, st, ws, wsb, 0)) std::exit(6);
      std::vector<T> h1(nq), h2(nq);
      CK(cudaMemcpy(h1.data(), O, nq * 2, cudaMemcpyDeviceToHost));
      if (!apa::prefill(Q, rs.k, rs.v, O, ps, eps, ws, wsb, 0)) std::exit(7);
      CK(cudaMemcpy(h2.data(), O, nq * 2, cudaMemcpyDeviceToHost));
      double d = 0, a = 0, bb = 0;
      size_t sd = 0;
      for (size_t i = 0; i < nq; ++i) {
        const double x = __half2float(h1[i]), y = __half2float(h2[i]);
        d += x * y, a += x * x, bb += y * y;
        sd += __half_as_ushort(h1[i]) != __half_as_ushort(h2[i]);
      }
      std::printf("stale-cache redo vs fresh prefill: cos %.6f, %zu of %zu elements differ\n",
                  d / std::sqrt(a * bb), sd, nq);
    }
    CK(cudaFree(sb));
  }
  // APA_B2=1: batch of 2 (batch 1 = V negated) vs B = 1 at the last eps: O[0] must equal the B = 1 output and
  // O[1] its negation, bitwise (sign-symmetric E2M1 / f16, same K stats and hot masks).
  if (std::getenv("APA_B2")) {
    const float eps = epss.back();
    const size_t ke = (size_t)kv * nkv * hd;  // K / V elements per batch (APA_KV may cut the dump)
    T *Q2, *K2, *V2, *O2;
    CK(cudaMalloc(&Q2, 2 * nq * 2));
    CK(cudaMalloc(&K2, 2 * ke * 2));
    CK(cudaMalloc(&V2, 2 * ke * 2));
    CK(cudaMalloc(&O2, 2 * nq * 2));
    std::vector<T> hvn(ke);
    for (size_t i = 0; i < ke; ++i) hvn[i] = __ushort_as_half(__half_as_ushort(hv[i]) ^ 0x8000u);
    for (int bb = 0; bb < 2; ++bb) {
      CK(cudaMemcpy(Q2 + bb * nq, Q, nq * 2, cudaMemcpyDeviceToDevice));
      CK(cudaMemcpy(K2 + bb * ke, K, ke * 2, cudaMemcpyDeviceToDevice));
    }
    CK(cudaMemcpy(V2, V, ke * 2, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(V2 + ke, hvn.data(), ke * 2, cudaMemcpyHostToDevice));
    apa::Problem p2 = p;
    p2.B = 2;
    const size_t wsb2 = apa::workspace_bytes(p2);
    void* ws2;
    CK(cudaMalloc(&ws2, wsb2));
    if (!apa::prefill(Q, K, V, O, p, eps, ws, wsb, 0) || !apa::prefill(Q2, K2, V2, O2, p2, eps, ws2, wsb2, 0))
      std::exit(10);
    std::vector<T> h1(nq), h2(2 * nq);
    CK(cudaMemcpy(h1.data(), O, nq * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2.data(), O2, 2 * nq * 2, cudaMemcpyDeviceToHost));
    size_t d0 = 0, d1 = 0, z1 = 0;  // z1: +0 in both batches (exact cancellation rounds to +0 either way)
    for (size_t i = 0; i < nq; ++i) {
      const unsigned a = __half_as_ushort(h1[i]), x = __half_as_ushort(h2[nq + i]), y = a ^ 0x8000u;
      d0 += __half_as_ushort(h2[i]) != a;
      const bool zero = ((x | y) & 0x7fffu) == 0;
      z1 += x != y && zero;
      if (x != y && !zero && d1++ == 0)
        std::printf("batch 1 first diff at %zu: %g vs -(%g)\n", i, __half2float(h2[nq + i]), __half2float(h1[i]));
    }
    std::printf("batch 2 eps %.0e: batch 0 vs B=1 %zu, batch 1 vs -(B=1) %zu of %zu elements differ (+-0: %zu)\n", eps,
                d0, d1, nq, z1);
    CK(cudaFree(Q2));
    CK(cudaFree(K2));
    CK(cudaFree(V2));
    CK(cudaFree(O2));
    CK(cudaFree(ws2));
  }
  CK(cudaFree(ws));

  return 0;
}
