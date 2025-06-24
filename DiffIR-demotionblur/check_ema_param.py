import torch

ckpt_path = '/home/ubuntu/data_sota_disk/model_space/diffIR/Deblurring-DiffIRS1.pth'  # 修改为你自己的路径

ckpt = torch.load(ckpt_path, map_location='cpu')

# 打印所有的 key
print("Checkpoint keys:", ckpt.keys())

# 检查是否有 'params_ema'
if 'params_ema' in ckpt:
    print("✅ 包含 'params_ema'")
else:
    print("❌ 不包含 'params_ema'")
