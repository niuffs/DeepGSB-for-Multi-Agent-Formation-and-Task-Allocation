#!/bin/bash
# ====================================================
# PINN Experiment Batch
# ====================================================

DEVICE="${DEVICE:-cuda:0}"
PYTHON="python"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR" || { echo "ERROR: cannot cd to $BASE_DIR"; exit 1; }

DATE_TAG=$(date +%Y-%m-%d)

echo "============================================="
echo " PINN Experiment Batch -- $DATE_TAG"
echo " Device: $DEVICE"
echo "============================================="
echo ""

run_experiment() {
    local NAME=$1
    local DIM=$2
    local EXTRA="$3"

    # Scale iterations and batch size with dimension
    local ITERS=5000
    local BATCH=500
    if [ "$DIM" -ge 50 ]; then
        ITERS=30000; BATCH=3000
    elif [ "$DIM" -ge 20 ]; then
        ITERS=20000; BATCH=1500
    elif [ "$DIM" -ge 10 ]; then
        ITERS=10000; BATCH=1000
    elif [ "$DIM" -ge 3 ]; then
        ITERS=8000; BATCH=800
    fi

    echo "[$(date '+%H:%M:%S')] Starting: $NAME d=$DIM iters=$ITERS bs=$BATCH $EXTRA"

    $PYTHON -u main_pinn.py \
        --problem-name "$NAME" \
        --dim "$DIM" \
        --iterations "$ITERS" \
        --batch-size "$BATCH" \
        --device "$DEVICE" \
        $EXTRA && RC=0 || RC=$?

    echo ""
}

# ====================================================================
# Gauss2Gauss_Obs3  --  d=3, 20, 50
# ====================================================================
echo "--- Gauss2Gauss_Obs3 d=3 -------------------------"
run_experiment "Gauss2Gauss_Obs3" 3

echo "--- Gauss2Gauss_Obs3 d=20 ------------------------"
run_experiment "Gauss2Gauss_Obs3" 20

echo "--- Gauss2Gauss_Obs3 d=50 ------------------------"
run_experiment "Gauss2Gauss_Obs3" 50

echo "=========================================="
echo " All PINN experiments completed!"
echo " Date: $(date)"
echo "=========================================="
