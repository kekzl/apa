#!/bin/sh
# Build paper/apa.pdf in a TeX Live container. Log: paper/build/apa.log
# IMAGE: any TeX Live image with latexmk (default texlive/texlive:latest-small).
set -e
cd "$(dirname "$0")"
mkdir -p build
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/w -w /w "${IMAGE:-texlive/texlive:latest-small}" \
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build apa.tex > build/latexmk.out 2>&1 ||
  { grep -E "^!|Error|not found" build/apa.log build/latexmk.out | head -20; exit 1; }
cp build/apa.pdf apa.pdf
grep -E "Warning" build/apa.log | grep -v "Font shape" | head -10 || true
