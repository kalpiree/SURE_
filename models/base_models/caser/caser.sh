#!/bin/bash

DATA_DIR="./phased_data"
CKPT_DIR="./ckpt_caser"
PHASES=5

mkdir -p $CKPT_DIR

for ((i=0; i<$PHASES; i++)); do
  echo "=================="
  echo "🚀 Phase $i — Caser"
  echo "=================="

  TRAIN_FILE="$DATA_DIR/train_phase${i}.txt"
  EVAL_FILE="$DATA_DIR/eval_phase${i}.csv"
  CKPT_SUBDIR="$CKPT_DIR/phase${i}"

  INIT_MODEL=""
  if [ $i -ne 0 ]; then
    INIT_MODEL="$CKPT_DIR/phase$((i-1))/model.pt"
    echo "🔁 Using checkpoint from previous phase: $INIT_MODEL"
  fi

  python main.py \
    --train_root=$TRAIN_FILE \
    --test_root=$EVAL_FILE \
    --train_dir=$CKPT_SUBDIR \
    --phase=$i \
    --n_iter=50 \
    --batch_size=512 \
    --learning_rate=1e-3 \
    --l2=1e-6 \
    --neg_samples=3 \
    --use_cuda=True \
    ${INIT_MODEL:+--resume=$INIT_MODEL}
done
