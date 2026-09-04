#!/usr/bin/env bash
# Copyright 2026, Sirius Contributors.
# Licensed under the Apache License, Version 2.0. See LICENSE at the repo root.
#
# Low-overhead Nsight Systems profiles of pinned_staging_bench.
#
# Only the measured passes are captured (cudaProfilerStart/Stop bracket them in
# the benchmark), CPU sampling / context-switch / backtrace collection are off,
# and only CUDA API + GPU memcpy activity + the benchmark's NVTX ranges are
# traced. Each configuration is also run once *without* nsys so the overhead can
# be quantified from the two CSV lines.
#
#   pixi run bench/pinned_staging/profile.sh
#
# Environment knobs:
#   TOTAL      dataset size per pass       (default 32G)
#   ITERS      measured passes             (default 3)
#   OUT_DIR    where reports go            (default build/bench/pinned_staging/nsys)
#   NSYS       nsys binary                 (default /usr/local/cuda/bin/nsys, else PATH)
#   CPU_SAMPLE also take one CPU-sampled profile of the best staged config (default 1)
#   ONLY       regex; only profile configurations whose label matches (appends to overhead.csv)
#   SKIP_MAIN  1 = skip the main matrix (CPU-sampled run only)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/bench/pinned_staging}"
BIN="$BUILD_DIR/pinned_staging_bench"
OUT_DIR="${OUT_DIR:-$BUILD_DIR/nsys}"
TOTAL="${TOTAL:-32G}"
ITERS="${ITERS:-3}"
CPU_SAMPLE="${CPU_SAMPLE:-1}"
if [[ -z "${NSYS:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nsys ]]; then NSYS=/usr/local/cuda/bin/nsys; else NSYS=nsys; fi
fi

if [[ ! -x "$BIN" ]]; then
  cmake -S "$REPO_ROOT/bench/pinned_staging" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR"
fi
mkdir -p "$OUT_DIR"

# Low-overhead trace: CUDA runtime API + GPU memcpy activity + NVTX only.
NSYS_FLAGS=(
  --trace=cuda,nvtx
  --sample=none
  --cpuctxsw=none
  --backtrace=none
  --cuda-memory-usage=false
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --stats=false
  --force-overwrite=true
)

# Serialize GPU access with other sessions on the box (see run_sweep.sh).
GPU_LOCK="${GPU_LOCK:-/tmp/sirius-pinned-bench.gpu.lock}"

OVERHEAD_CSV="$OUT_DIR/overhead.csv"
if [[ ! -f "$OVERHEAD_CSV" || ( "${SKIP_MAIN:-0}" != "1" && -z "${ONLY:-}" ) ]]; then
  echo "config,label,plain_gbps,profiled_gbps,overhead_pct" >"$OVERHEAD_CSV"
fi

# gbps <stdout>: median_gbps is column 10 of the benchmark's CSV line. nsys also
# prints a "Generated: <path>" line to stdout, so pick the benchmark line only.
gbps() {
  grep -m1 -E '^(pinned|pageable|memcpy|staged),' <<<"$1" | tr -d '\r\n' | awk -F, '{print $10}'
}

profile() {
  local label="$1"
  shift
  if [[ -n "${ONLY:-}" && ! "$label" =~ ${ONLY} ]]; then return; fi
  local rep="$OUT_DIR/$label"
  echo "=== $label: $*" >&2

  local plain profiled
  plain=$(flock -w 7200 "$GPU_LOCK" \
    "$BIN" --csv --iters "$ITERS" --total-bytes "$TOTAL" "$@" 2>"$OUT_DIR/$label.plain.log")
  profiled=$(flock -w 7200 "$GPU_LOCK" "$NSYS" profile "${NSYS_FLAGS[@]}" -o "$rep" -- \
    "$BIN" --csv --nvtx --iters "$ITERS" --total-bytes "$TOTAL" "$@" 2>"$OUT_DIR/$label.nsys.log")

  local p q
  p=$(gbps "$plain")
  q=$(gbps "$profiled")
  awk -v c="$*" -v l="$label" -v p="$p" -v q="$q" 'BEGIN {
    printf "%s,%s,%s,%s,%.2f\n", "\"" c "\"", l, p, q, (p - q) / p * 100
  }' >>"$OVERHEAD_CSV"

  # Text reports next to the .nsys-rep; also produces $rep.sqlite for analyze_nsys.py.
  "$NSYS" stats --force-export=true --format csv \
    --report cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum \
    --output "$rep" "$rep.nsys-rep" >/dev/null 2>"$OUT_DIR/$label.stats.log" || true
  python3 "$REPO_ROOT/bench/pinned_staging/analyze_nsys.py" "$rep.sqlite" >"$rep.analysis.txt" 2>&1 || true
}

if [[ "${SKIP_MAIN:-0}" != "1" ]]; then
  profile pinned-64M-t8 --mode pinned --chunk-bytes 64M --threads 8
  profile pageable-64M-t8 --mode pageable --chunk-bytes 64M --threads 8
  profile pinned-1M-t8 --mode pinned --chunk-bytes 1M --threads 8
  profile pageable-1M-t8 --mode pageable --chunk-bytes 1M --threads 8
  profile staged-64M-t8-s2 --mode staged --chunk-bytes 64M --threads 8 --slots 2
  profile staged-64M-t32-s2 --mode staged --chunk-bytes 64M --threads 32 --slots 2
  profile staged-64M-t32-s1 --mode staged --chunk-bytes 64M --threads 32 --slots 1
  # Shared pool: deep asynchronous DMA queue (up to --slots copies queued).
  profile staged-pool-64M-t8-s64-k8 --mode staged --chunk-bytes 64M --threads 8 \
    --sched pool --slots 64 --streams 8
  profile staged-pool-64M-t32-s128-k32 --mode staged --chunk-bytes 64M --threads 32 \
    --sched pool --slots 128 --streams 32
fi

if [[ "$CPU_SAMPLE" == "1" ]]; then
  # One CPU-sampled run to name the hot host functions (memcpy variant). Leaf IP
  # only (no backtraces) and a long sampling period (cycles between samples, so
  # larger = fewer samples) keep the overhead low; otherwise CUDA-trace-only.
  label=staged-64M-t32-s2-cpusample
  rep="$OUT_DIR/$label"
  echo "=== $label (CPU sampling)" >&2
  profiled=$(flock -w 7200 "$GPU_LOCK" "$NSYS" profile --trace=cuda,nvtx --sample=process-tree \
    --sampling-period="${SAMPLING_PERIOD:-16000000}" \
    --cpuctxsw=none --backtrace=none --cuda-memory-usage=false \
    --capture-range=cudaProfilerApi --capture-range-end=stop --stats=false \
    --force-overwrite=true -o "$rep" -- \
    "$BIN" --csv --nvtx --iters "$ITERS" --total-bytes "$TOTAL" \
    --mode staged --chunk-bytes 64M --threads 32 --slots 2 2>"$OUT_DIR/$label.nsys.log")
  plain=$(grep '"--mode staged --chunk-bytes 64M --threads 32 --slots 2"' "$OVERHEAD_CSV" | head -1 | awk -F, '{print $3}')
  if [[ -z "$plain" ]]; then
    plain=$(gbps "$(flock -w 7200 "$GPU_LOCK" "$BIN" --csv --iters "$ITERS" --total-bytes "$TOTAL" \
      --mode staged --chunk-bytes 64M --threads 32 --slots 2 2>"$OUT_DIR/$label.plain.log")")
  fi
  awk -v l="$label" -v p="$plain" -v q="$(gbps "$profiled")" 'BEGIN {
    printf "\"--mode staged --chunk-bytes 64M --threads 32 --slots 2 (cpu sampling)\",%s,%s,%s,%.2f\n", l, p, q, (p - q) / p * 100
  }' >>"$OVERHEAD_CSV"
  "$NSYS" stats --force-export=true --format csv \
    --report cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum \
    --output "$rep" "$rep.nsys-rep" >/dev/null 2>"$OUT_DIR/$label.stats.log" || true
  python3 "$REPO_ROOT/bench/pinned_staging/analyze_nsys.py" "$rep.sqlite" >"$rep.analysis.txt" 2>&1 || true
fi

echo >&2
echo "reports in $OUT_DIR" >&2
column -s, -t <"$OVERHEAD_CSV" >&2
