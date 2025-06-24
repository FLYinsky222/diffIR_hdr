# HDR任务训练指南

## 概述

本指南介绍了专为LDR到HDR图像恢复任务设计的训练流程。新的训练流程移除了原始去模糊任务中的渐进式训练复杂性，提供了更直接、高效的HDR训练方案。

## 主要改进

### 🔄 训练流程简化

**原始流程 (train_pipeline.py)**:
- ✅ 渐进式训练：不同迭代阶段使用不同的batch size和patch size
- ✅ 复杂的调度策略：`mini_batch_sizes`, `iters`, `gt_sizes`
- ✅ 适合去模糊任务的多阶段训练

**HDR流程 (train_pipeline_hdr.py)**:
- ✅ 简化训练：固定的batch size和patch size
- ✅ 专注HDR任务：移除不必要的复杂性
- ✅ 支持dgain信息：更好地处理HDR特有数据

### 📊 学习率调度器优化

**新的CosineAnnealingWarmupRestarts调度器**:
```yaml
scheduler:
  type: CosineAnnealingWarmupRestarts
  first_cycle_steps: 50000      # 第一个周期的步数
  cycle_mult: 1.5               # 周期倍增因子
  max_lr: 2e-4                  # 最大学习率
  min_lr: 1e-7                  # 最小学习率
  warmup_steps: 2000            # warmup步数
  gamma: 0.8                    # 每个周期的学习率衰减因子
```

**优势**:
- 🚀 更快的训练启动 (2k步warmup)
- 🔄 更频繁的重启 (5万步周期)
- 📉 长期学习率衰减防止过拟合
- 🎯 精确的学习率控制

## 文件结构

```
DiffIR-demotionblur/
├── train_pipeline.py          # 原始去模糊训练流程
├── train_pipeline_hdr.py      # HDR专用训练流程
├── train_hdr.py               # HDR训练启动脚本
├── options/
│   ├── train_DiffIRS1.yml     # 去模糊训练配置
│   └── train_DiffIRs1_hdr.yml # HDR训练配置
└── DiffIR/models/
    └── lr_scheduler.py        # 学习率调度器实现
```

## 使用方法

### 1. 基本训练
```bash
python train_hdr.py -opt options/train_DiffIRs1_hdr.yml
```

### 2. 分布式训练
```bash
python -m torch.distributed.launch --nproc_per_node=2 train_hdr.py -opt options/train_DiffIRs1_hdr.yml --launcher pytorch
```

### 3. 恢复训练
```bash
python train_hdr.py -opt options/train_DiffIRs1_hdr.yml --auto_resume
```

## 配置文件对比

### 原始配置 (train_DiffIRS1.yml)
```yaml
datasets:
  train:
    batch_size_per_gpu: 8
    mini_batch_sizes: [8,4,2,2,1,1]
    iters: [92000,80000,38000,33000,33000,24000]
    gt_size: 512
    gt_sizes: [192, 256,320,384,464,512]

scheduler:
  type: CosineAnnealingRestartCyclicLR
  periods: [92000, 208000]
  restart_weights: [1,1]
  eta_mins: [0.000285,0.000001]
```

### HDR配置 (train_DiffIRs1_hdr.yml)
```yaml
datasets:
  train:
    type: LDR_HDR_PairedDataset  # HDR专用数据集
    batch_size_per_gpu: 8        # 固定批次大小
    gt_size: 512                 # 固定图像尺寸
    dataroot_dgain: ...          # dgain信息路径

scheduler:
  type: CosineAnnealingWarmupRestarts  # 新的调度器
  first_cycle_steps: 50000
  cycle_mult: 1.5
  max_lr: 2e-4
  min_lr: 1e-7
  warmup_steps: 2000
  gamma: 0.8
```

## 核心改进详解

### 1. 数据处理简化
**原始流程**:
```python
# 复杂的渐进式训练逻辑
j = ((current_iter>groups) !=True).nonzero()[0]
mini_gt_size = mini_gt_sizes[bs_j]
mini_batch_size = mini_batch_sizes[bs_j]
# 动态调整batch size和patch size
```

**HDR流程**:
```python
# 简化的数据处理
lq = train_data['lq']  # 低质量图像（LDR）
gt = train_data['gt']  # 真实图像（HDR）
dgain_info = train_data.get('dgain', None)  # HDR特有信息
```

### 2. 训练逻辑优化
- ✅ 移除复杂的阶段性训练
- ✅ 专注于HDR任务的特殊需求
- ✅ 支持dgain信息传递
- ✅ 简化的日志输出

### 3. 性能优化
- 🚀 减少训练代码复杂度
- 📊 更稳定的训练过程
- 🎯 更好的HDR任务适配性
- 💪 更强的可扩展性

## 监控和调试

### 日志输出
HDR训练流程提供清晰的中文日志输出：
```
HDR训练统计信息:
    训练图像数量: 1000
    数据集放大比例: 1
    每GPU批次大小: 8
    世界大小(GPU数量): 1
    每轮次所需迭代数: 125
    总轮次: 2400; 总迭代数: 300000.

HDR训练配置:
  - 批次大小: 8
  - 图像尺寸: 512x512
  - 总迭代数: 300000
```

### TensorBoard监控
训练过程会自动记录：
- 学习率变化曲线
- 损失函数变化
- 验证指标（PSNR等）
- 训练图像样本

## 故障排除

### 常见问题

**Q: 训练速度比原始流程慢？**
A: HDR流程去除了渐进式训练，在相同硬件条件下应该更快。检查数据加载和网络配置。

**Q: 内存不足？**
A: 降低`batch_size_per_gpu`或`gt_size`，或使用梯度累积。

**Q: 学习率调度器报错？**
A: 确保使用了更新后的`lr_scheduler.py`文件。

### 性能调优建议

1. **数据加载优化**:
   ```yaml
   num_worker_per_gpu: 8-16  # 根据CPU核心数调整
   prefetch_mode: cuda       # 使用GPU预取
   pin_memory: true
   ```

2. **内存优化**:
   ```yaml
   batch_size_per_gpu: 4-8   # 根据GPU内存调整
   gt_size: 256-512          # 根据需求调整
   ```

3. **训练稳定性**:
   ```yaml
   warmup_iter: 2000         # 适度的warmup
   ema_decay: 0.999          # EMA权重更新
   ```

## 总结

HDR训练流程提供了：
- ✅ **简化的训练逻辑**: 专注HDR任务，移除不必要复杂性
- ✅ **优化的学习率调度**: 更适合HDR任务的调度策略
- ✅ **更好的可维护性**: 清晰的代码结构和中文注释
- ✅ **增强的调试支持**: 详细的日志输出和错误处理

这个新的训练流程特别适合LDR到HDR的图像恢复任务，能够提供更稳定、高效的训练体验。 