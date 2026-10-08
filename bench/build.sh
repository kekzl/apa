#!/bin/sh
# Build build/bench_apa (CUDA 13 toolchain image with nvcc, sm_120a). Log: build/build.log
# IMAGE: any image with nvcc >= 12.9 (default imp:toolchain). Extra flags: NVFLAGS="-DAPA_EXACT_MAX".
set -e
cd "$(dirname "$0")/.."
mkdir -p build
docker run --rm -e NVFLAGS="$NVFLAGS" -u "$(id -u):$(id -g)" -v "$PWD":/w -w /w "${IMAGE:-imp:toolchain}" \
  sh -c "nvcc -O3 $NVFLAGS -std=c++20 -gencode arch=compute_120a,code=sm_120a -lineinfo -Xptxas -v bench/bench_apa.cu -o build/bench_apa" \
  > build/build.log 2>&1 || { grep -E "error" build/build.log | head -20; exit 1; }
grep -A2 -E "Compiling entry.*apa_kernel" build/build.log | grep -E "registers|spill" | head -6
