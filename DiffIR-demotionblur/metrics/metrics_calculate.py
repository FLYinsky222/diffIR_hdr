import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1, img2):
    """计算两张图片的PSNR值"""
    # 确保图片数据类型为float32，范围在[0,1]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    # 计算PSNR
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """计算两张图片的SSIM值"""
    # 确保图片数据类型为float32，范围在[0,1]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 如果是彩色图片，需要指定multichannel=True
    if len(img1.shape) == 3:
        ssim = structural_similarity(img1, img2, multichannel=True, channel_axis=2, data_range=1.0)
    else:
        ssim = structural_similarity(img1, img2, data_range=1.0)
    
    return ssim