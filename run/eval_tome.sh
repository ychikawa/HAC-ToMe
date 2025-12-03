#!/bin/bash

DATA_PATH=/path/to/your/imagenet
# deit_tiny, deit_small, deit_base, augreg_tiny, augreg_small, augreg_base, sam_base, lvvit_t, lvvit_s, lvvit_m, lvvit_l, vit_base_patch16_mae, vit_large_patch16_mae, vit_huge_patch14_mae
MODEL=deit_small
ALGO=tome
RESULT_DIR=./results_${ALGO}/
R=20

mkdir -p "$RESULT_DIR"

python main.py \
	--eval \
	--model $MODEL \
	--data_path  $DATA_PATH \
	--task_type [1,0,0] \
	--algo $ALGO \
	--r $R \
	--benchmark \
	>> "${RESULT_DIR}/${MODEL}_R${R}.txt"
