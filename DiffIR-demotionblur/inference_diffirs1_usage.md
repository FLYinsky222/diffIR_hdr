# DiffIRS1 推理脚本使用指南

## 🚀 概述

修改后的 `inference_diffirs1.py` 脚本现在支持**自动检测和使用EMA权重**，解决了训练时开启EMA但推理时未正确加载EMA权重的问题。

## 📋 主要改进

### 1. **自动EMA权重检测**
- 优先加载 `params_ema`（EMA权重）
- 如果EMA权重不存在或加载失败，自动回退到 `params`（普通权重）
- 提供详细的权重加载日志

### 2. **灵活的权重选择**
- `--use_ema`: 强制使用EMA权重
- `--no_ema`: 强制跳过EMA权重，直接使用普通权重

### 3. **权重差异分析**
- 当同时存在EMA和普通权重时，自动比较两者差异
- 提供权重统计信息，帮助判断EMA效果

### 4. **快速权重检查**
- `--check-weights`: 只检查模型权重，不运行推理

## 🔧 使用方法

### 基础推理（自动选择权重）
```bash
python inference_diffirs1.py \
    --model_path /path/to/your/model.pth \
    --input /path/to/input/images \
    --gt /path/to/ground/truth \
    --output results/output
```

### 强制使用EMA权重
```bash
python inference_diffirs1.py \
    --model_path /path/to/your/model.pth \
    --input /path/to/input/images \
    --gt /path/to/ground/truth \
    --output results/output \
    --use_ema
```

### 强制跳过EMA权重
```bash
python inference_diffirs1.py \
    --model_path /path/to/your/model.pth \
    --input /path/to/input/images \
    --gt /path/to/ground/truth \
    --output results/output \
    --no_ema
```

### 快速检查模型权重
```bash
python inference_diffirs1.py --check-weights --model_path /path/to/your/model.pth
```

## 📊 输出示例

### 权重检查输出
```
🔍 Checking model weights in: /path/to/model.pth

Available keys in checkpoint:
  ✅ params: dict with 284 items
  ✅ params_ema: dict with 284 items
  ✅ optimizer: dict with 6 items
  ✅ schedulers: list
  ✅ epoch: <class 'int'>
  ✅ iter: <class 'int'>

🎯 EMA weights found! (params_ema)
   EMA权重包含 284 个参数
🎯 Regular weights found! (params)
   普通权重包含 284 个参数
```

### 权重加载输出
```
Loading model from: /path/to/model.pth
Available keys in checkpoint:
  params: dict with 284 items
  params_ema: dict with 284 items
  optimizer: dict with 6 items

自动检测：优先使用EMA权重（如果可用）
Found EMA weights, loading params_ema...
✅ Successfully loaded EMA weights

📊 Comparing regular weights vs EMA weights:
  Common parameters: 284
  encoder.conv_first.weight: relative diff = 0.005234
  encoder.down1.0.norm1.weight: relative diff = 0.003891
  encoder.down1.0.attn.to_q.weight: relative diff = 0.004567
  encoder.down1.0.attn.to_k.weight: relative diff = 0.003234
  encoder.down1.0.attn.to_v.weight: relative diff = 0.004123
  📈 Overall relative difference: 0.004210
  ✨ EMA and regular weights are very similar

Model loaded successfully and moved to cuda:0
```

## 🎯 重要说明

### EMA的作用
- **EMA（指数移动平均）**是一种在训练过程中维护模型参数平滑版本的技术
- EMA权重通常比普通训练权重有更好的泛化性能
- 在推理时使用EMA权重可以获得更稳定和更好的结果

### 配置文件中的EMA设置
在训练配置中通常会看到：
```yaml
train:
  ema_decay: 0.999  # EMA衰减系数
```

### 权重选择建议
1. **默认情况**：让脚本自动选择（优先EMA）
2. **性能对比**：可以分别使用 `--use_ema` 和 `--no_ema` 对比结果
3. **调试模式**：使用 `--check-weights` 确认模型权重内容

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   ```
   ❌ Failed to load EMA weights: ...
   ```
   - 检查模型文件是否完整
   - 确认模型架构是否匹配

2. **找不到EMA权重**
   ```
   ⚠️ No EMA weights found (params_ema not present)
   ```
   - 确认训练时是否开启了EMA
   - 检查保存的checkpoint是否包含EMA权重

3. **权重键名不匹配**
   - 使用 `--check-weights` 查看实际的键名
   - 可能需要根据实际情况调整 `load_model_weights` 函数

## 📈 性能对比

建议进行以下对比测试：

```bash
# 使用EMA权重
python inference_diffirs1.py --use_ema --output results/ema

# 使用普通权重  
python inference_diffirs1.py --no_ema --output results/regular

# 比较结果
# 通常EMA权重会产生更好的PSNR/SSIM指标
```

## 🔧 扩展功能

脚本还包含以下增强功能：
- 自动检测GT文件的不同扩展名
- 更好的错误处理和日志输出
- 详细的处理进度显示
- 完整的权重分析和比较功能 