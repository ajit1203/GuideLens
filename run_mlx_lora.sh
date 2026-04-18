#!/usr/bin/env bash
set -e

python -m mlx_vlm.lora \
  --model-path mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --dataset data/processed/mlx_qwen_ds \
  --split train \
  --batch-size 1 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --gradient-accumulation-steps 4 \
  --grad-checkpoint \
  --train-on-completions \
  --output-path checkpoints/mlx_qwen25vl_adapter.safetensors