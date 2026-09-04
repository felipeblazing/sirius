#!/usr/bin/env bash
# Copyright 2026, Sirius Contributors.
# Licensed under the Apache License, Version 2.0. See LICENSE at the repo root.
#
# Full buffer-size x thread-count grid for the three strategies plotted by
# plot_results.py: pinned source, pageable source (direct copy), and buffered
# (pageable -> preallocated pinned ring with 2 slots/thread -> device).
#
#   pixi run bench/pinned_staging/run_grid.sh
#   pixi run python bench/pinned_staging/plot_results.py <csv> --threads 1,8,32 -o grid.svg
#
# Environment knobs:
#   TOTAL      dataset size per pass                    (default 32G)
#   ITERS      measured passes per configuration       (default 3)
#   CHUNKS     buffer sizes                            (default "1M 2M 4M 8M 16M 32M 64M")
#   THREADS    thread counts                           (default "1 2 4 8 16 32")
#   BUILD_DIR  where to build                          (default build/bench/pinned_staging)
#   OUT        CSV output path                         (default $BUILD_DIR/grid-<host>-<date>.csv)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/bench/pinned_staging}"
BIN="$BUILD_DIR/pinned_staging_bench"
TOTAL="${TOTAL:-32G}"
ITERS="${ITERS:-3}"
CHUNKS="${CHUNKS:-1M 2M 4M 8M 16M 32M 64M}"
THREADS="${THREADS:-1 2 4 8 16 32}"
OUT="${OUT:-$BUILD_DIR/grid-$(hostname -s)-$(date +%Y%m%d-%H%M%S).csv}"

if [[ ! -x "$BIN" ]]; then
  cmake -S "$REPO_ROOT/bench/pinned_staging" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR"
fi
mkdir -p "$(dirname "$OUT")"
echo "writing $OUT" >&2

# Serialize GPU access with other sessions on the box: these numbers are host-
# memory-bandwidth-bound, so two concurrent runs corrupt each other.
GPU_LOCK="${GPU_LOCK:-/tmp/sirius-pinned-bench.gpu.lock}"
# RUN_PREFIX, e.g. "numactl --cpunodebind=0 --membind=0", is put in front of
# every benchmark invocation (word-split on purpose).
RUN_PREFIX="${RUN_PREFIX:-}"
run() {
  echo "--- $*" >&2
  # shellcheck disable=SC2086
  flock -w 7200 "$GPU_LOCK" $RUN_PREFIX "$BIN" --csv --iters "$ITERS" --total-bytes "$TOTAL" \
    "$@" >>"$OUT"
}

"$BIN" --csv-header --mode alloc --total-bytes 1M >"$OUT"
for t in $THREADS; do
  for c in $CHUNKS; do
    run --mode pinned --chunk-bytes "$c" --threads "$t"
    run --mode pageable --chunk-bytes "$c" --threads "$t"
    run --mode staged --chunk-bytes "$c" --threads "$t" --slots 2
  done
done

echo "results: $OUT" >&2
