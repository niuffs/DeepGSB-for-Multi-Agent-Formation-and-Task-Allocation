#!/bin/bash
# ====================================================
# MFG-Net Experiment Batch (simplified: 3 scenarios)
# ====================================================

DEVICE="${DEVICE:-cuda:1}"
PYTHON="python"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR" || { echo "ERROR: cannot cd to $BASE_DIR"; exit 1; }

DATE_TAG=$(date +%Y-%m-%d)

echo "============================================="
echo " MFG-Net Experiment Batch -- $DATE_TAG"
echo " Device: $DEVICE"
echo "============================================="
echo ""

run_experiment() {
    local NAME=$1
    local DIM=$2
    local EXTRA="$3"

    local M=64
    local NT=4
    local STEPPER="rk4"
    if [ "$DIM" -gt 100 ]; then M=256; NT=8; STEPPER="rk1"
    elif [ "$DIM" -gt 50 ]; then M=128; NT=6; STEPPER="rk1"
    elif [ "$DIM" -gt 20 ]; then M=96; NT=6
    elif [ "$DIM" -gt 3 ]; then M=96; NT=4
    fi

    local BATCH_SIZE=256
    if [ "$DIM" -gt 100 ]; then BATCH_SIZE=64
    elif [ "$DIM" -gt 50 ]; then BATCH_SIZE=128
    elif [ "$DIM" -gt 20 ]; then BATCH_SIZE=192
    fi

    echo "[$(date '+%H:%M:%S')] Starting: $NAME d=$DIM bs=$BATCH_SIZE m=$M nt=$NT stepper=$STEPPER $EXTRA"

    $PYTHON -u main_mfg_net.py \
        --env "$NAME" \
        --dim "$DIM" \
        --batch-size "$BATCH_SIZE" \
        --m "$M" \
        --nt "$NT" \
        --stepper "$STEPPER" \
        --num-repeats 5 \
        --device "$DEVICE" \
        --plot-freq 50 \
        --print-freq 20 \
        $EXTRA && RC=0 || RC=$?

    echo ""
}

# ====================================================================
# (1) Gauss2Rings_Obs1  --  d=2, 3
# ====================================================================
echo "--- (1.1) Gauss2Rings_Obs1 ----------------------"
run_experiment "Gauss2Rings_Obs1" 2  "--max-iters 300"
run_experiment "Gauss2Rings_Obs1" 3  "--max-iters 300"
echo "--- (1.2) Rings2Gauss_Obs1 ----------------------"
run_experiment "Rings2Gauss_Obs1" 2  "--max-iters 500"
run_experiment "Rings2Gauss_Obs1" 3  "--max-iters 500"

# ====================================================================
# (2) Gauss2Gauss_Obs3  --  d=20, 50
# ====================================================================
echo "--- (2) Gauss2Gauss_Obs3 ----------------------"
run_experiment "Gauss2Gauss_Obs3" 20 "--max-iters 1000"
run_experiment "Gauss2Gauss_Obs3" 50 "--max-iters 1000"

echo "=========================================="
echo " All MFG-Net experiments completed!"
echo " Date: $(date)"
echo "=========================================="
