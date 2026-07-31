#!/bin/bash
set -uo pipefail
cd /home/nvidia/felipe/sirius/.claude/worktrees/combo-prefetch-comp
WT=$(pwd)
while true; do
  used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
  [ -z "$used" ] || [ "$used" -lt 1024 ] && break
  echo "$(date +%H:%M:%S) GPU busy — waiting"; sleep 60
done
echo "=== $(date +%H:%M:%S) QUENT RUN: tuned + GPU-compressed pins ==="
SIRIUS_PRE_SQL="SET pin_table_compression = true; SET pin_table_input_compression_plan_dir = '$WT/bench_out/plans_gpu_facts'" \
SIRIUS_PIN_TIER_LINEITEM=gpu SIRIUS_PIN_TIER_ORDERS=gpu \
SIRIUS_PIN_TIER_PART=gpu SIRIUS_PIN_TIER_CUSTOMER=gpu SIRIUS_PIN_TIER_SUPPLIER=gpu \
SIRIUS_PIN_TIER_NATION=gpu SIRIUS_PIN_TIER_REGION=gpu \
  pixi run python test/tpch_performance/performance_test.py \
    --input /home/nvidia/tpch_parquet_sf1000 --data-source parquet \
    --engine gpu --mode grouped --iterations 3 --pin host \
    --config bench/sirius_p8_pf3_quent.yaml --output bench_out/combined --name armZq_quent
echo "=== $(date +%H:%M:%S) QUENT RUN COMPLETE ==="
