#!/bin/bash

# === CONFIGURATION ===
DATASET="sasrec"
SUBDATASET="goodreads"
# ALPHAS="0.05 0.07 0.1 0.12 0.15"
ALPHAS="0.1"
ETAS="0.5"
GAMMA="2.0"
FREEZE="--freeze_inference"
OUTPUT_DIR="outputs"
MAX_PRED_SET_SIZE="51"
BASE_UTILS="recall=1.0"
# recall=0.67,ndcg=0.6,
# === EXECUTION ===
# Debug command here:
echo python main_.py --dataset "$DATASET" --subdataset "$SUBDATASET" --alphas $ALPHAS --etas $ETAS --gamma $GAMMA $FREEZE --output_dir "$OUTPUT_DIR" --max_pred_set_size "$MAX_PRED_SET_SIZE" --base_utilities "$BASE_UTILS"
time python main_.py \
    --dataset "$DATASET" \
    --subdataset "$SUBDATASET" \
    --alphas $ALPHAS \
    --etas $ETAS \
    --gamma $GAMMA \
    $FREEZE \
    --output_dir "$OUTPUT_DIR" \
    --max_pred_set_size "$MAX_PRED_SET_SIZE" \
    --base_utilities "$BASE_UTILS"
