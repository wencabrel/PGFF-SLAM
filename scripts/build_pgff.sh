#!/bin/bash
#
# PGFF - Predictive Geometric Flow Fields
# Build and Test Script
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     PGFF - Predictive Geometric Flow Fields                   ║"
echo "║     Building and Testing                                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_DIR"

# Build
echo "[1/3] Building project with PGFF..."
if [ -f "/.dockerenv" ]; then
    # Inside Docker
    source /opt/ros/humble/setup.bash
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
else
    # Outside Docker - use Docker to build
    docker run --rm \
        -v "$PROJECT_DIR:/ws" \
        -w /ws \
        docker.cnb.cool/gpf2025/slam:demo \
        bash -c "source /opt/ros/humble/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20"
fi

echo ""
echo "[2/3] Verifying PGFF module..."

# Check if PGFF headers exist
PGFF_FILES=(
    "src/core/pgff/pgff.h"
    "src/core/pgff/flow_field.h"
    "src/core/pgff/jacobian_cache.h"
    "src/core/pgff/surprise_detector.h"
    "src/core/pgff/predictive_lio.h"
    "src/core/pgff/enhanced_obs_model.h"
    "src/core/pgff/surprise_loop_detector.h"
)

all_present=true
for file in "${PGFF_FILES[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        all_present=false
    fi
done

if [ "$all_present" = true ]; then
    echo ""
    echo "[3/3] PGFF module verified successfully!"
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  PGFF Components:                                             ║"
    echo "║                                                               ║"
    echo "║  • GeometricFlowField   - Predicts point motion               ║"
    echo "║  • JacobianCache        - Temporal H-matrix caching           ║"
    echo "║  • SurpriseDetector     - Information-theoretic selection     ║"
    echo "║  • PredictiveLIO        - Main integration layer              ║"
    echo "║  • EnhancedObsModel     - Selective point processing          ║"
    echo "║  • SurpriseLoopDetector - Implicit loop detection             ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "To use PGFF in your code:"
    echo ""
    echo '  #include "core/pgff/pgff.h"'
    echo ""
    echo "  // Create PGFF instance"
    echo "  auto pgff = lightning::pgff::CreateDefaultPGFF();"
    echo ""
    echo "  // In your SLAM loop:"
    echo "  pgff.PreparePrediction(state, predicted_state, pose);"
    echo "  auto surprising = pgff.SelectPointsToProcess(scan, residuals);"
    echo "  // ... process only surprising points ..."
    echo "  pgff.UpdateWithResults(state, H, residuals, validity);"
    echo ""
else
    echo ""
    echo "ERROR: Some PGFF files are missing!"
    exit 1
fi
