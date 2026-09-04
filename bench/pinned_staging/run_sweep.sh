#!/usr/bin/env bash
# Copyright 2026, Sirius Contributors.
# Licensed under the Apache License, Version 2.0. See LICENSE at the repo root.
#
# Builds pinned_staging_bench (if needed) and runs the parameter sweep that
# backs bench/pinned_staging/README.md. Run from the repo root inside the pixi
# environment so nvcc/cmake/ninja resolve:
#
#   pixi run bench/pinned_staging/run_sweep.sh              # full sweep
#   TOTAL=8G pixi run bench/pinned_staging/run_sweep.sh     # quicker
#
# Environment knobs:
#   TOTAL      dataset size per copy pass            (default 32G)
#   ITERS      measured passes per configuration     (default 3)
#   BUILD_DIR  where to build                        (default build/bench/pinned_staging)
#   OUT        CSV output path                       (default $BUILD_DIR/sweep-<host>-<date>.csv)
#   ALLOC_SIZES  sizes for the startup-cost probe    (default "8G 32G 128G")
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/bench/pinned_staging"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/bench/pinned_staging}"
BIN="$BUILD_DIR/pinned_staging_bench"
TOTAL="${TOTAL:-32G}"
ITERS="${ITERS:-3}"
ALLOC_SIZES="${ALLOC_SIZES:-8G 32G 128G}"
OUT="${OUT:-$BUILD_DIR/sweep-$(hostname -s)-$(date +%Y%m%d-%H%M%S).csv}"

if [[ ! -x "$BIN" ]]; then
  cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
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
  # Each invocation appends one CSV line; human-readable progress goes to stderr.
  echo "--- $*" >&2
  # shellcheck disable=SC2086
  flock -w 7200 "$GPU_LOCK" $RUN_PREFIX "$BIN" --csv --iters "$ITERS" "$@" >>"$OUT"
}

"$BIN" --csv-header --mode alloc --total-bytes 1M >"$OUT"

# 1. Startup cost of pinning: what Sirius pays today, per pinned GiB.
for kind in hostalloc register; do
  for sz in $ALLOC_SIZES; do
    run --mode alloc --pinned-alloc "$kind" --total-bytes "$sz"
  done
done

# 2. Ceiling: source already pinned. 1M matches cucascade's default block size.
for chunk in 1M 16M 64M 256M; do
  for t in 1 2 4 8 16; do
    run --mode pinned --total-bytes "$TOTAL" --chunk-bytes "$chunk" --threads "$t"
  done
done

# 3. Baseline: cudaMemcpyAsync straight from pageable memory (driver staging).
for chunk in 1M 16M 64M 256M; do
  for t in 1 2 4 8 16 32; do
    run --mode pageable --total-bytes "$TOTAL" --chunk-bytes "$chunk" --threads "$t"
  done
done

# 4. CPU-side ceiling for stage A alone (pageable -> pinned memcpy, no GPU).
for t in 1 2 4 8 16 32 64; do
  run --mode memcpy --total-bytes "$TOTAL" --chunk-bytes 64M --threads "$t" --slots 2
done

# 5. The hypothesis: pageable -> pinned ring -> device.
for chunk in 16M 64M 256M; do
  for slots in 1 2 4; do
    for t in 2 4 8 16 32 64; do
      run --mode staged --total-bytes "$TOTAL" --chunk-bytes "$chunk" --threads "$t" \
        --slots "$slots"
    done
  done
done

# 5b. Shared-pool scheduling: any producer fills any free slot, DMAs are issued
# asynchronously on shared streams and a reaper thread recycles slots, so up to
# --slots copies can be queued on the GPU regardless of producer count.
for t in 8 16 32; do
  for slots in 32 128; do
    for streams in 8 32; do
      run --mode staged --total-bytes "$TOTAL" --chunk-bytes 64M --threads "$t" \
        --sched pool --slots "$slots" --streams "$streams"
    done
  done
done
run --mode staged --total-bytes "$TOTAL" --chunk-bytes 16M --threads 32 --sched pool \
  --slots 256 --streams 32
run --mode staged --total-bytes "$TOTAL" --chunk-bytes 64M --threads 64 --sched pool \
  --slots 128 --streams 32

# 6. Knobs that might move the CPU side: core pinning, transparent huge pages.
run --mode staged --total-bytes "$TOTAL" --chunk-bytes 64M --threads 32 --slots 2 --pin-cpus
run --mode staged --total-bytes "$TOTAL" --chunk-bytes 64M --threads 32 --slots 2 --huge
run --mode pageable --total-bytes "$TOTAL" --chunk-bytes 64M --threads 8 --huge

# 7. Correctness check on one representative staged configuration.
run --mode staged --total-bytes "$TOTAL" --chunk-bytes 64M --threads 16 --slots 2 --verify

echo >&2
echo "results: $OUT" >&2
column -s, -t <"$OUT" >&2
