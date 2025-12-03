#!/bin/bash

DATA_PATH=/path/to/your/imagenet
# deit_tiny, deit_small, deit_base, augreg_tiny, augreg_small, augreg_base, sam_base, lvvit_t, lvvit_s, lvvit_m, lvvit_l, vit_base_patch16_mae, vit_large_patch16_mae, vit_huge_patch14_mae
MODEL=deit_small
ALGO=tome_LHAC
RESULT_DIR=./results_${ALGO}/
R=20

export CUDA_VISIBLE_DEVICES=1,2,3,4

mkdir -p "$RESULT_DIR"

torchrun --nproc_per_node=4 -- main.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --batch-size 256 --epochs 1 \
    --output_dir ./output \
    --task_type [1,0,0] \
    --task_weight [1,0,0] \
    --algo $ALGO \
    --r $R \
    --modular \
    >> "${RESULT_DIR}/${MODEL}_R${R}.txt"