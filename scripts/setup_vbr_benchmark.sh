#!/bin/bash
# Setup VBR SLAM Benchmark for evaluating PGFF

set -e

echo "=== Setting up VBR SLAM Benchmark ==="

# Compile the benchmark
cd /home/wen/Light-3D/vbr-slam-benchmark
echo "Compiling VBR benchmark..."
g++ -o vbr_benchmark vbr_benchmark.cpp -I /usr/include/eigen3 -O3 -std=c++17

echo "✓ VBR benchmark compiled successfully"
echo ""
echo "Usage:"
echo "  ./vbr_benchmark /path/to/vbr_gt/ /path/to/estimates/ [--plot]"
echo ""
echo "For PGFF evaluation:"
echo "  1. Run SLAM on VBR sequences to generate trajectory_*.txt files"
echo "  2. Organize trajectory files in a directory matching VBR sequence names"
echo "  3. Run benchmark to get ATE/RPE metrics"
echo ""
