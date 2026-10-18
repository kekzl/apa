#!/bin/sh
# Test matrix on a GPU. DUMPS: dir of dump *.bin (all used), OUT: logs (build/test), IMAGE (imp:toolchain),
# REF: optional bench binary in build/ of an earlier commit with the same kernel; outputs must match bitwise.
# Checks per dump: determinism, batch 2, paged vs flat, chunked vs requant at restats, stale cache, KV window vs
# every tile ahead, pass 2 per window segment vs at the end / after segment 0 (bitwise), all-pairs accuracy (no NaN,
# no negative cos); forced prep overflow redo vs exact stats (window
# redo on dumps over 12352 keys); sliding window (APA_SWA, flat and every tile, q_offset > 0) vs a windowed FP32 ref,
# window >= kv vs causal, paged = flat, chunked = requant at restats, window -1 declined by every entry point. Synthetic dumps: build/bench_apa syn:n,kv,nh,nkv,seed[,hd] with APA_SAVE=file.
# NOBUILD=1: reuse build/test_* (build them first without a GPU). Exit 1 on any failed check.
set -u
cd "$(dirname "$0")/.."
DUMPS=${DUMPS:-../imp-ra2/build-dev}
OUT=${OUT:-build/test}
case "$OUT" in /*) echo "OUT must be relative to the repo (APA_OUT files go to /w/\$OUT)"; exit 1 ;; esac
IMAGE=${IMAGE:-imp:toolchain}
mkdir -p "$OUT"
fail=0
check() {  # name, then the command that must succeed
  n=$1; shift
  if "$@"; then echo "ok   $n"; else echo "FAIL $n"; fail=1; fi
}
build() {  # name nvflags
  NVFLAGS="$2" sh bench/build.sh > /dev/null 2>&1 && ! grep -qE "warning|error" build/build.log &&
    cp build/bench_apa "build/test_$1"
}
run() {  # bin dump log env...
  b=$1 d=$2 l=$3; shift 3
  docker run --rm --gpus all "$@" -v "$PWD":/w -v "$(realpath "$DUMPS")":/d:ro -w /w "$IMAGE" "./build/$b" "/d/$d.bin" \
    > "$OUT/$l.log" 2>&1
}
has() { grep -qE "$2" "$OUT/$1.log"; }
none() { ! grep -qE "$2" "$OUT/$1.log"; }

for v in "def:" "ps0:-DAPA_PREP_SAMPLE=0" "pp2:-DAPA_PREP_SAMPLE=1000000" "pov:-DAPA_PREP_HEADROOM=0.015625f" \
         "dbg:-DAPA_DBG" "f32o:-DAPA_P2_F32O=1" "xm:-DAPA_EXACT_MAX" "ns3:-DAPA_WIN_SLOTS=3" \
         "w0:-DAPA_WIN=0" "s0:-DAPA_P2_SEG=0" "e1:-DAPA_P2_EARLY=1"; do
  [ -n "${NOBUILD:-}" ] || check "build ${v%%:*}" build "${v%%:*}" "${v#*:}"
done
for f in "$DUMPS"/*.bin; do
  d=$(basename "$f" .bin)
  run test_def "$d" "$d" -e APA_FULL=1 -e APA_EPS=0.005 -e APA_DET=1 -e APA_B2=1 -e APA_PAGED=16 -e APA_CHUNKS=2 \
    -e APA_WIN=32,64,128
  check "$d run" has "$d" "^chunks incr"
  check "$d window = every tile" none "$d" "^window S .* [1-9][0-9]* of [0-9]+ output elements differ"
  check "$d determinism" has "$d" "^determinism: 0 of"
  check "$d batch 2" has "$d" "batch 0 vs B=1 0, batch 1 vs -\(B=1\) 0 of"
  check "$d paged = flat" has "$d" "^paged vs flat: 0 of"
  check "$d chunked = requant at restats" none "$d" "bitwise equal 0 of"
  check "$d stale cache redo" none "$d" "^stale-cache redo.*, [1-9][0-9]* of"
  check "$d no NaN / cos < 0" none "$d" "nan|NaN|cos<0 [1-9]"
  grep -E "^full" "$OUT/$d.log" | cut -c1-120
  run test_pov "$d" "${d}_pov" -e APA_EPS=0.005 -e APA_OUT=/w/$OUT/${d}_pov
  run test_pp2 "$d" "${d}_pp2" -e APA_EPS=0.005 -e APA_OUT=/w/$OUT/${d}_pp2
  check "$d forced overflow redo = exact stats" cmp -s "$OUT/${d}_pov_0.005000" "$OUT/${d}_pp2_0.005000"
  run test_def "$d" "${d}_dout" -e APA_EPS=0.005 -e APA_OUT=/w/$OUT/${d}_dout
  for sv in s0 e1; do  # pass 2 per segment (state carried) vs all at the end / after segment 0 only: bitwise
    run test_$sv "$d" "${d}_$sv" -e APA_EPS=0.005 -e APA_OUT=/w/$OUT/${d}_$sv
    check "$d pass 2 per segment = $sv" cmp -s "$OUT/${d}_dout_0.005000" "$OUT/${d}_${sv}_0.005000"
  done
  run test_ns3 "$d" "${d}_ns3" -e APA_EPS=0.005 -e APA_WIN=32,64,96
  check "$d window, 3 slots run" has "${d}_ns3" "^window S 96"
  check "$d window, 3 slots = every tile" none "${d}_ns3" "^window S .* [1-9][0-9]* of [0-9]+ output elements differ"
  for sv in def w0; do
    run "test_$sv" "$d" "${d}_swa_$sv" -e APA_EPS=0.005 -e APA_SWA=1024,4096,1048576
    check "$d sliding window ($sv) run" has "${d}_swa_$sv" "^swa 4096 eps 5e-03"
    check "$d sliding window ($sv) exact = windowed ref" none "${d}_swa_$sv" "^swa [0-9]+ eps -1e\+00: .* bad [1-9]"
    check "$d sliding window ($sv) >= kv = causal" has "${d}_swa_$sv" "vs causal: 0 of"
    check "$d sliding window ($sv) gate" has "${d}_swa_$sv" "^swa gate: window -1 declined by 3 of 3"
    check "$d sliding window ($sv) paged run" has "${d}_swa_$sv" "^swa 4096 eps 5e-03 paged"
    check "$d sliding window ($sv) paged = flat" none "${d}_swa_$sv" "^swa .* paged bs [0-9]+ vs flat: [1-9]"
    check "$d sliding window ($sv) chunked = requant at restats" none "${d}_swa_$sv" "bitwise 0 of"
    check "$d sliding window ($sv) chunked exact = windowed ref" none "${d}_swa_$sv" \
      "^swa [0-9]+ eps -1e\+00 chunks .* bad [1-9]"
  done
  run test_def "$d" "${d}_swa_sp" -e APA_EPS=0.005 -e APA_SPARSE=8192 -e APA_SWA=1024,4096
  check "$d sliding window, q_offset > 0 exact" none "${d}_swa_sp" "^swa [0-9]+ eps -1e\+00: .* bad [1-9]"
  check "$d sliding window, q_offset > 0 run" has "${d}_swa_sp" "^swa 4096 eps 5e-03"
  check "$d sliding window, q_offset > 0 paged = flat" none "${d}_swa_sp" "^swa .* paged bs [0-9]+ vs flat: [1-9]"
  if [ -n "${REF:-}" ] && [ "$(od -An -tu4 -j16 -N4 "$f" | tr -d " ")" = 128 ]; then  # REF builds: hd 128 only
    run "$REF" "$d" "${d}_ref" -e APA_EPS=0.005 -e APA_OUT=/w/$OUT/${d}_ref
    run test_def "$d" "${d}_cur" -e APA_EPS=0.005 -e APA_OUT=/w/$OUT/${d}_cur
    check "$d default = $REF" cmp -s "$OUT/${d}_ref_0.005000" "$OUT/${d}_cur_0.005000"
  fi
done
[ $fail -eq 0 ] && echo "ALL OK" || echo "FAILED"
exit $fail
