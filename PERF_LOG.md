# PERF_LOG

Append-only. Each entry: date, commit, build flags, command, verbatim bench lines (`grep`, never retyped).
GPU: RTX 5090, image imp:toolchain (CUDA 13.4.2). Dumps: ~/github.com/kekzl/imp-ra2/build-dev/lc_*.bin.

## 2026-10-08 sweep, APA 0.2.0 (main c74647a), default build

Script: per dump bench default eps list, then `APA_EPS` 0.002 / 0.005 / 0.02; `APA_KV` 8192..65536 x eps {-1, 0.005}
on lc_122880_1. Filter: `grep -E "^##|^dump|^apa eps"`.

```
## lc_122880_0 default
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps   -1e+00: cos 0.996468 min 0.977271  hot 100.0 % p2cta 100.0 %  attn 11.806 ms (pass2 12.479) 259.7 TOPS  (prep 0.733 ms)
apa eps    1e-03: cos 0.999275 min 0.978584  hot  37.0 % p2cta  44.3 %  attn 8.587 ms (pass2 6.327) 357.1 TOPS  (prep 0.733 ms)
apa eps    3e-03: cos 0.999320 min 0.937773  hot  17.0 % p2cta  21.1 %  attn 5.450 ms (pass2 2.928) 562.7 TOPS  (prep 0.733 ms)
apa eps    1e-02: cos 0.998450 min 0.857188  hot   4.6 % p2cta   6.2 %  attn 3.999 ms (pass2 0.919) 766.8 TOPS  (prep 0.733 ms)
## lc_122880_0 eps 0.002
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps    2e-03: cos 0.999463 min 0.952552  hot  23.0 % p2cta  28.0 %  attn 6.310 ms (pass2 3.963) 486.0 TOPS  (prep 0.732 ms)
## lc_122880_0 eps 0.005
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps    5e-03: cos 0.999000 min 0.898838  hot  10.4 % p2cta  13.5 %  attn 4.545 ms (pass2 1.810) 674.7 TOPS  (prep 0.732 ms)
## lc_122880_0 eps 0.02
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps    2e-02: cos 0.997993 min 0.817225  hot   2.0 % p2cta   2.8 %  attn 3.631 ms (pass2 0.439) 844.5 TOPS  (prep 0.733 ms)
## lc_122880_1 default
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps   -1e+00: cos 0.999634 min 0.988660  hot 100.0 % p2cta 100.0 %  attn 11.910 ms (pass2 12.479) 257.5 TOPS  (prep 0.732 ms)
apa eps    1e-03: cos 0.999771 min 0.982954  hot  28.2 % p2cta  45.0 %  attn 7.479 ms (pass2 5.291) 410.0 TOPS  (prep 0.732 ms)
apa eps    3e-03: cos 0.999714 min 0.973783  hot  14.1 % p2cta  25.2 %  attn 5.450 ms (pass2 2.931) 562.7 TOPS  (prep 0.732 ms)
apa eps    1e-02: cos 0.999432 min 0.968365  hot   5.5 % p2cta  10.8 %  attn 4.183 ms (pass2 1.313) 733.1 TOPS  (prep 0.732 ms)
## lc_122880_1 eps 0.002
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps    2e-03: cos 0.999754 min 0.975364  hot  18.7 % p2cta  31.9 %  attn 6.223 ms (pass2 3.695) 492.8 TOPS  (prep 0.731 ms)
## lc_122880_1 eps 0.005
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps    5e-03: cos 0.999626 min 0.970893  hot   9.7 % p2cta  18.0 %  attn 4.620 ms (pass2 2.123) 663.8 TOPS  (prep 0.732 ms)
## lc_122880_1 eps 0.02
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps    2e-02: cos 0.999181 min 0.966923  hot   2.9 % p2cta   6.0 %  attn 3.833 ms (pass2 0.774) 800.1 TOPS  (prep 0.735 ms)
## lc_122880_2 default
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps   -1e+00: cos 0.999596 min 0.876635  hot 100.0 % p2cta 100.0 %  attn 11.818 ms (pass2 12.398) 259.5 TOPS  (prep 0.732 ms)
apa eps    1e-03: cos 0.998623 min -0.400998  hot  18.6 % p2cta  29.1 %  attn 6.859 ms (pass2 4.000) 447.1 TOPS  (prep 0.732 ms)
apa eps    3e-03: cos 0.997910 min -0.439857  hot   9.1 % p2cta  16.2 %  attn 5.274 ms (pass2 2.557) 581.5 TOPS  (prep 0.732 ms)
apa eps    1e-02: cos 0.996548 min -0.444615  hot   3.2 % p2cta   6.6 %  attn 4.247 ms (pass2 1.202) 722.1 TOPS  (prep 0.732 ms)
## lc_122880_2 eps 0.002
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps    2e-03: cos 0.998212 min -0.439857  hot  12.1 % p2cta  20.5 %  attn 5.711 ms (pass2 3.031) 537.0 TOPS  (prep 0.733 ms)
## lc_122880_2 eps 0.005
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps    5e-03: cos 0.997437 min -0.444615  hot   6.0 % p2cta  11.4 %  attn 4.757 ms (pass2 1.921) 644.7 TOPS  (prep 0.733 ms)
## lc_122880_2 eps 0.02
dump n 2048 kv 122880 off 120832 nh 24 nkv 8
apa eps    2e-02: cos 0.995494 min -0.444615  hot   1.7 % p2cta   3.6 %  attn 3.844 ms (pass2 0.744) 797.8 TOPS  (prep 0.734 ms)
## lc_32768_0 default
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps   -1e+00: cos 0.999895 min 0.999458  hot 100.0 % p2cta 100.0 %  attn 3.153 ms (pass2 3.187) 253.3 TOPS  (prep 0.162 ms)
apa eps    1e-03: cos 0.999890 min 0.995742  hot  92.6 % p2cta  96.2 %  attn 3.562 ms (pass2 3.133) 224.2 TOPS  (prep 0.162 ms)
apa eps    3e-03: cos 0.999779 min 0.968032  hot  62.4 % p2cta  72.4 %  attn 2.878 ms (pass2 2.369) 277.6 TOPS  (prep 0.162 ms)
apa eps    1e-02: cos 0.998506 min 0.695481  hot  17.4 % p2cta  23.7 %  attn 1.529 ms (pass2 0.901) 522.4 TOPS  (prep 0.162 ms)
## lc_32768_0 eps 0.002
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    2e-03: cos 0.999865 min 0.985541  hot  77.5 % p2cta  85.3 %  attn 3.236 ms (pass2 2.764) 246.9 TOPS  (prep 0.161 ms)
## lc_32768_0 eps 0.005
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    5e-03: cos 0.999438 min 0.879942  hot  39.7 % p2cta  50.3 %  attn 2.263 ms (pass2 1.676) 353.0 TOPS  (prep 0.163 ms)
## lc_32768_0 eps 0.02
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    2e-02: cos 0.997643 min 0.605165  hot   7.1 % p2cta  10.2 %  attn 1.174 ms (pass2 0.448) 680.4 TOPS  (prep 0.161 ms)
## lc_32768_1 default
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps   -1e+00: cos 0.999951 min 0.997074  hot 100.0 % p2cta 100.0 %  attn 3.169 ms (pass2 3.199) 252.1 TOPS  (prep 0.163 ms)
apa eps    1e-03: cos 0.999938 min 0.996972  hot  62.0 % p2cta  81.3 %  attn 3.020 ms (pass2 2.450) 264.5 TOPS  (prep 0.163 ms)
apa eps    3e-03: cos 0.999873 min 0.991455  hot  34.4 % p2cta  55.0 %  attn 2.297 ms (pass2 1.655) 347.7 TOPS  (prep 0.163 ms)
apa eps    1e-02: cos 0.999634 min 0.972697  hot  13.3 % p2cta  24.9 %  attn 1.516 ms (pass2 0.810) 526.9 TOPS  (prep 0.163 ms)
## lc_32768_1 eps 0.002
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    2e-03: cos 0.999909 min 0.994720  hot  44.4 % p2cta  65.9 %  attn 2.561 ms (pass2 1.976) 311.9 TOPS  (prep 0.161 ms)
## lc_32768_1 eps 0.005
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    5e-03: cos 0.999797 min 0.984021  hot  23.7 % p2cta  41.0 %  attn 1.865 ms (pass2 1.252) 428.3 TOPS  (prep 0.162 ms)
## lc_32768_1 eps 0.02
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    2e-02: cos 0.999417 min 0.964543  hot   7.2 % p2cta  14.1 %  attn 1.261 ms (pass2 0.525) 633.6 TOPS  (prep 0.164 ms)
## lc_32768_2 default
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps   -1e+00: cos 0.999902 min 0.986612  hot 100.0 % p2cta 100.0 %  attn 3.141 ms (pass2 3.158) 254.4 TOPS  (prep 0.163 ms)
apa eps    1e-03: cos 0.999416 min 0.337745  hot  45.5 % p2cta  58.9 %  attn 2.712 ms (pass2 2.045) 294.5 TOPS  (prep 0.163 ms)
apa eps    3e-03: cos 0.998895 min 0.100775  hot  25.6 % p2cta  38.6 %  attn 2.151 ms (pass2 1.383) 371.4 TOPS  (prep 0.163 ms)
apa eps    1e-02: cos 0.997978 min 0.093914  hot   9.6 % p2cta  17.8 %  attn 1.523 ms (pass2 0.854) 524.6 TOPS  (prep 0.163 ms)
## lc_32768_2 eps 0.002
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    2e-03: cos 0.999113 min 0.182524  hot  32.8 % p2cta  46.3 %  attn 2.242 ms (pass2 1.626) 356.3 TOPS  (prep 0.160 ms)
## lc_32768_2 eps 0.005
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    5e-03: cos 0.998569 min 0.100775  hot  17.6 % p2cta  29.1 %  attn 1.842 ms (pass2 1.104) 433.7 TOPS  (prep 0.164 ms)
## lc_32768_2 eps 0.02
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    2e-02: cos 0.997147 min 0.093914  hot   4.9 % p2cta   9.9 %  attn 1.270 ms (pass2 0.576) 628.9 TOPS  (prep 0.163 ms)
## lc_122880_1 kv 8192 eps -1
dump n 2048 kv 8192 off 6144 nh 24 nkv 8
apa eps   -1e+00: cos 0.999948 min 0.999759  hot 100.0 % p2cta 100.0 %  attn 0.760 ms (pass2 0.767) 237.3 TOPS  (prep 0.044 ms)
## lc_122880_1 kv 8192 eps 0.005
dump n 2048 kv 8192 off 6144 nh 24 nkv 8
apa eps    5e-03: cos 0.999811 min 0.989181  hot  35.8 % p2cta  60.0 %  attn 0.580 ms (pass2 0.422) 311.0 TOPS  (prep 0.048 ms)
## lc_122880_1 kv 16384 eps -1
dump n 2048 kv 16384 off 14336 nh 24 nkv 8
apa eps   -1e+00: cos 0.999817 min 0.998786  hot 100.0 % p2cta 100.0 %  attn 1.572 ms (pass2 1.563) 245.9 TOPS  (prep 0.087 ms)
## lc_122880_1 kv 16384 eps 0.005
dump n 2048 kv 16384 off 14336 nh 24 nkv 8
apa eps    5e-03: cos 0.999403 min 0.990550  hot  31.5 % p2cta  55.0 %  attn 1.102 ms (pass2 0.798) 350.8 TOPS  (prep 0.054 ms)
## lc_122880_1 kv 32768 eps -1
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps   -1e+00: cos 0.999470 min 0.994855  hot 100.0 % p2cta 100.0 %  attn 3.135 ms (pass2 3.157) 254.8 TOPS  (prep 0.163 ms)
## lc_122880_1 kv 32768 eps 0.005
dump n 2048 kv 32768 off 30720 nh 24 nkv 8
apa eps    5e-03: cos 0.998390 min 0.962551  hot  21.4 % p2cta  41.8 %  attn 1.919 ms (pass2 1.215) 416.2 TOPS  (prep 0.163 ms)
## lc_122880_1 kv 65536 eps -1
dump n 2048 kv 65536 off 63488 nh 24 nkv 8
apa eps   -1e+00: cos 0.998949 min 0.974452  hot 100.0 % p2cta 100.0 %  attn 6.318 ms (pass2 6.527) 257.0 TOPS  (prep 0.383 ms)
## lc_122880_1 kv 65536 eps 0.005
dump n 2048 kv 65536 off 63488 nh 24 nkv 8
apa eps    5e-03: cos 0.998524 min 0.866539  hot  17.8 % p2cta  33.8 %  attn 3.312 ms (pass2 1.892) 490.3 TOPS  (prep 0.382 ms)
```

Other 2026-10-08 runs (0.2.0 vs 0.3.0 A/B, eps sweep with APA_DIAG, kv scaling 0.3.0, APA_P2_F32O A/B) exist only in a
session transcript, not as files; their numbers are in paper 4963e56 and are re-measured in Phase 1.
