#!/usr/bin/env bash

# 使用torchrun的HDR训练脚本（推荐）
# 更现代、更稳定的分布式训练启动方式

echo "🚀 启动HDR训练 - 使用torchrun"
echo "=================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    DiffIR/train_hdr.py \
    -opt options/train_DiffIRs1_hdr.yml \
    --launcher pytorch

echo "✅ 训练完成"
