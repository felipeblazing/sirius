#!/bin/bash
set -uo pipefail
cd /home/nvidia/felipe/sirius/.claude/worktrees/combo-prefetch-comp
WT=$(pwd)
CFG=bench/sirius_p8_pf3.yaml
OUT=bench_out/combined
RUNNER=test/tpch_performance/performance_test.py
mkdir -p $OUT
ARGS=(--input /home/nvidia/tpch_parquet_sf1000 --data-source parquet
      --engine gpu --mode grouped --iterations 3 --pin host)

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
    [ -z "$used" ] || [ "$used" -lt 1024 ] && break
    echo "$(date +%H:%M:%S) GPU busy (${used} MiB) — waiting 60s"; sleep 60
  done
}

echo "=== $(date +%H:%M:%S) ARM T: tuned stack alone (p8/blk64/pf3, host raw pins) ==="
wait_for_gpu
pixi run python "$RUNNER" "${ARGS[@]}" --config "$CFG" --output "$OUT" --name armT_tuned_only

echo "=== $(date +%H:%M:%S) ARM Z: tuned stack + GPU-compressed pins ==="
wait_for_gpu
SIRIUS_PRE_SQL="SET pin_table_compression = true; SET pin_table_input_compression_plan_dir = '$WT/bench_out/plans_gpu_facts'" \
SIRIUS_PIN_TIER_LINEITEM=gpu SIRIUS_PIN_TIER_ORDERS=gpu \
SIRIUS_PIN_TIER_PART=gpu SIRIUS_PIN_TIER_CUSTOMER=gpu SIRIUS_PIN_TIER_SUPPLIER=gpu \
SIRIUS_PIN_TIER_NATION=gpu SIRIUS_PIN_TIER_REGION=gpu \
  pixi run python "$RUNNER" "${ARGS[@]}" --config "$CFG" --output "$OUT" --name armZ_tuned_plus_comp

echo "=== $(date +%H:%M:%S) COMBINED ARMS COMPLETE ==="
