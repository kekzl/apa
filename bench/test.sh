#!/bin/sh
# Test matrix on a GPU. DUMPS: dir of dump *.bin (all used), OUT: logs (build/test), IMAGE (imp:toolchain),
# REF: optional bench binary in build/ of an earlier commit with the same kernel; outputs must match bitwise.
# Checks per dump: determinism, batch 2, paged vs flat, chunked vs requant at restats, stale cache, all-pairs
# accuracy (no NaN, no negative cos); forced prep overflow redo vs exact stats. Exit 1 on any failed check.
set -u
cd "$(dirname "$0")/.."
DUMPS=${DUMPS:-../imp-ra2/build-dev}
OUT=${OUT:-build/test}
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
         "dbg:-DAPA_DBG" "f32o:-DAPA_P2_F32O=1" "xm:-DAPA_EXACT_MAX"; do
  check "build ${v%%:*}" build "${v%%:*}" "${v#*:}"
done
for f in "$DUMPS"/*.bin; do
  d=$(basename "$f" .bin)
  run test_def "$d" "$d" -e APA_FULL=1 -e APA_EPS=0.005 -e APA_DET=1 -e APA_B2=1 -e APA_PAGED=16 -e APA_CHUNKS=2
  check "$d run" has "$d" "^chunks incr"
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
  if [ -n "${REF:-}" ]; then
    run "$REF" "$d" "${d}_ref" -e APA_EPS=0.005 -e APA_OUT=/w/$OUT/${d}_ref
    run test_def "$d" "${d}_cur" -e APA_EPS=0.005 -e APA_OUT=/w/$OUT/${d}_cur
    check "$d default = $REF" cmp -s "$OUT/${d}_ref_0.005000" "$OUT/${d}_cur_0.005000"
  fi
done
[ $fail -eq 0 ] && echo "ALL OK" || echo "FAILED"
exit $fail
