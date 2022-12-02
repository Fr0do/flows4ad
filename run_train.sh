#!/bin/bash

DATASET=6_cardio

python train.py \
    \
    --dataset_name ${DATASET} \
    --data_dir datasets/Classical \
    \
    --batch_size 64 \
    --num_workers 4 \
    \
    --num_flow_layers 16 \
    --num_mlp_layers 2 \
    --layer_norm \
    \
    --use_channel_wise_splits \
    --use_checkerboard_splits \
    \
    --lr 1e-3 \
    --seed 42 \
    --output_dir results/${DATASET} \
    \
    --plot_histogram \
    --log_wandb \