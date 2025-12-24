#!/bin/bash
# Compare PGFF-enhanced SLAM with baseline

WORKSPACE=/workspace
BAG_FILE=data/VBR/campus/ros2.db3
CONFIG_FILE=config/default_vbr_noui.yaml

echo "=== SLAM Comparison: PGFF vs Baseline ==="
echo ""

# Run PGFF-enhanced version
echo "Running PGFF-enhanced SLAM..."
timeout 300 ./bin/run_slam_offline -config $CONFIG_FILE -input_bag $BAG_FILE 2>&1 | tee /tmp/pgff_run.log

# Extract key metrics
echo ""
echo "=== Results Summary ==="
echo ""

# Count keyframes
KF_COUNT=$(grep -c "LIO: create kf" /tmp/pgff_run.log)
echo "Total Keyframes: $KF_COUNT"

# Count loop closures
LC_COUNT=$(grep -c "optimize finished, loops:" /tmp/pgff_run.log)
echo "Loop Closure Optimizations: $LC_COUNT"

# Get final loop count
FINAL_LOOPS=$(grep "optimize finished, loops:" /tmp/pgff_run.log | tail -1 | grep -oP 'loops: \K\d+')
echo "Final Loop Count: ${FINAL_LOOPS:-0}"

# Count PGFF high-weight events
HIGH_WEIGHT=$(grep "\[PGFF\]" /tmp/pgff_run.log | tail -20)
echo ""
echo "Last 5 PGFF reports:"
echo "$HIGH_WEIGHT" | tail -5

# Get surprise statistics
echo ""
echo "Surprise score samples:"
grep "surprise:" /tmp/pgff_run.log | awk -F'surprise: ' '{print $2}' | head -10

echo ""
echo "=== Comparison Complete ==="
