#!/bin/bash

DEVICE="${DEVICE:-cuda:0}"
PYTHON="python"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR" || { echo "ERROR: cannot cd to $BASE_DIR"; exit 1; }

COMMON="--num-itr 250 --seed 42"
DATE_TAG=$(date +%Y-%m-%d)

echo "============================================="
echo " DeepGSB Experiment Batch -- $DATE_TAG"
echo " Device: $DEVICE"
echo "============================================="
echo ""

run_experiment() {
    local NAME=$1
    local DIM=$2
    local EXTRA="$3"

    local SAMP_BS=5000
    if [ "$DIM" -gt 100 ]; then SAMP_BS=1000
    elif [ "$DIM" -gt 50 ]; then SAMP_BS=1500
    elif [ "$DIM" -gt 20 ]; then SAMP_BS=3000
    fi

    echo "[$(date '+%H:%M:%S')] Starting: $NAME d=$DIM stg=$EXTRA bs=$SAMP_BS"

    $PYTHON -u main_deepgsb.py \
        --problem-name "$NAME" \
        --x-dim-override "$DIM" \
        --device "$DEVICE" \
        --samp-bs "$SAMP_BS" \
        $COMMON $EXTRA \
        && RC=0 || RC=$?

    echo ""
}

# ====================================================================
# (1) Gauss2Ring_Obs1   --  d=2, 3
# ====================================================================
echo "--- (1) Gauss2Ring_Obs1 -----------------------"
run_experiment "Gauss2Ring_Obs1" 2  "--num-stage 10"
run_experiment "Gauss2Ring_Obs1" 3  "--num-stage 10"

# ====================================================================
# (2) Gauss2Rings_Obs1  --  d=2, 3
# ====================================================================
echo "--- (2) Gauss2Rings_Obs1 ----------------------"
run_experiment "Gauss2Rings_Obs1" 2  "--num-stage 20"
run_experiment "Gauss2Rings_Obs1" 3  "--num-stage 20"

# ====================================================================
# (3) MixGauss2SphereShell_Obs3  --  d=3, 20
# ====================================================================
echo "--- (3) MixGauss2SphereShell_Obs3 -------------"
run_experiment "MixGauss2SphereShell_Obs3" 10 "--num-stage 30"
run_experiment "MixGauss2SphereShell_Obs3" 20 "--num-stage 30"

# ====================================================================
# (4) Gauss2Gauss_Obs3  --  d=20, 50
# ====================================================================
echo "--- (4) Gauss2Gauss_Obs3 ----------------------"
run_experiment "Gauss2Gauss_Obs3" 20 "--num-stage 40"
run_experiment "Gauss2Gauss_Obs3" 50 "--num-stage 40"

echo "=========================================="
echo " All experiments completed!"
echo " Date: $(date)"
echo "=========================================="
