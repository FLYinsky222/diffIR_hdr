#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PU21 Python实现示例
演示如何使用PU21编码和metrics来评估HDR图像质量
"""

import numpy as np
import cv2
from metrics.pu21_encoder import PU21Encoder
from metrics.pu21_metrics import pu21_metric, pu21_encode_image
from scipy.ndimage import gaussian_filter

def load_hdr_image(file_path):
    """
    加载HDR图像
    """
    try:
        # 尝试使用OpenCV加载HDR图像
        hdr_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if hdr_image is None:
            raise ValueError(f"无法读取HDR文件：{file_path}")
        
        # 转换BGR到RGB
        if hdr_image.ndim == 3:
            hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
        
        return hdr_image.astype(np.float32)
    except Exception as e:
        print(f"加载HDR图像时出错：{e}")
        return None

def example_hdr_metrics():
    """
    HDR图像质量评估示例
    """
    print("=== HDR图像质量评估示例 ===")
    
    # 创建模拟HDR图像
    np.random.seed(42)
    I_ref = np.random.rand(128, 128, 3) * 1000  # 0-1000 cd/m^2
    
    # 设置显示器峰值亮度
    L_peak = 4000  # cd/m^2
    
    # 将图像映射到绝对亮度单位
    I_ref = I_ref / np.max(I_ref) * L_peak
    
    print(f"参考图像亮度范围: {I_ref.min():.2f} - {I_ref.max():.2f} cd/m^2")
    
    # 创建测试图像：添加高斯噪声
    noise_level = 0.2
    I_test_noise = np.maximum(
        I_ref + I_ref * np.random.randn(*I_ref.shape) * noise_level, 
        0.05
    )
    
    # 创建测试图像：添加模糊
    I_test_blur = gaussian_filter(I_ref, sigma=2.0)
    
    # 计算PU21 metrics
    print("\n计算PU21 metrics...")
    
    # PSNR
    psnr_noise = pu21_metric(I_test_noise, I_ref, 'PSNR')
    psnr_blur = pu21_metric(I_test_blur, I_ref, 'PSNR')
    
    # SSIM
    ssim_noise = pu21_metric(I_test_noise, I_ref, 'SSIM')
    ssim_blur = pu21_metric(I_test_blur, I_ref, 'SSIM')
    
    # MS-SSIM
    msssim_noise = pu21_metric(I_test_noise, I_ref, 'MSSSIM')
    msssim_blur = pu21_metric(I_test_blur, I_ref, 'MSSSIM')
    
    print(f"\n噪声图像 (噪声水平 {noise_level}):")
    print(f"  PSNR = {psnr_noise:.2f} dB")
    print(f"  SSIM = {ssim_noise:.4f}")
    print(f"  MS-SSIM = {msssim_noise:.4f}")
    
    print(f"\n模糊图像 (sigma=2.0):")
    print(f"  PSNR = {psnr_blur:.2f} dB")
    print(f"  SSIM = {ssim_blur:.4f}")
    print(f"  MS-SSIM = {msssim_blur:.4f}")

def example_pu21_encoding():
    """
    PU21编码示例
    """
    print("\n=== PU21编码示例 ===")
    
    # 创建PU21编码器
    pu21 = PU21Encoder('banding_glare')
    
    # 测试不同亮度值的编码
    luminance_values = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 4000.0])
    
    print("亮度值 (cd/m^2) -> PU21编码值:")
    for L in luminance_values:
        if L >= pu21.L_min and L <= pu21.L_max:
            encoded = pu21.encode(L)
            decoded = pu21.decode(encoded)
            print(f"  {L:8.1f} -> {encoded:8.2f} -> {decoded:8.2f} (误差: {abs(L-decoded):.6f})")
        else:
            print(f"  {L:8.1f} -> 超出有效范围")

def example_different_encodings():
    """
    不同PU21编码类型的比较
    """
    print("\n=== 不同PU21编码类型比较 ===")
    
    encoding_types = ['banding', 'banding_glare', 'peaks', 'peaks_glare']
    test_luminance = 100.0  # cd/m^2
    
    print(f"测试亮度值: {test_luminance} cd/m^2")
    print("编码类型 -> PU21值:")
    
    for enc_type in encoding_types:
        pu21 = PU21Encoder(enc_type)
        encoded = pu21.encode(test_luminance)
        print(f"  {enc_type:15s} -> {encoded:8.2f}")

def example_custom_metric():
    """
    自定义metric示例
    """
    print("\n=== 自定义Metric示例 ===")
    
    # 创建测试图像
    np.random.seed(42)
    I_ref = np.random.rand(64, 64, 3) * 1000
    I_test = I_ref + np.random.randn(*I_ref.shape) * 50
    
    # 定义自定义metric：简单的均方误差
    def custom_mse(img1, img2):
        return np.mean((img1 - img2) ** 2)
    
    # 使用自定义metric
    mse_value = pu21_metric(I_test, I_ref, custom_mse)
    
    print(f"自定义MSE metric: {mse_value:.2f}")

def example_tone_mapping_evaluation():
    """
    色调映射评估示例
    """
    print("\n=== 色调映射评估示例 ===")
    
    # 创建HDR图像
    np.random.seed(42)
    hdr_image = np.random.rand(64, 64, 3) * 5000  # 高动态范围
    
    # 模拟不同的色调映射算法
    def simple_tone_mapping(hdr, gamma=2.2):
        """简单的gamma色调映射"""
        normalized = hdr / np.max(hdr)
        return normalized ** (1.0 / gamma)
    
    def reinhard_tone_mapping(hdr):
        """Reinhard色调映射"""
        return hdr / (1 + hdr)
    
    # 创建参考LDR图像（假设这是"理想"的色调映射结果）
    ldr_reference = simple_tone_mapping(hdr_image, gamma=2.0) * 100  # 映射到100 cd/m^2
    
    # 测试不同的色调映射方法
    ldr_gamma = simple_tone_mapping(hdr_image, gamma=2.2) * 100
    ldr_reinhard = reinhard_tone_mapping(hdr_image / np.max(hdr_image)) * 100
    
    # 使用PU21评估质量
    psnr_gamma = pu21_metric(ldr_gamma, ldr_reference, 'PSNR')
    psnr_reinhard = pu21_metric(ldr_reinhard, ldr_reference, 'PSNR')
    
    ssim_gamma = pu21_metric(ldr_gamma, ldr_reference, 'SSIM')
    ssim_reinhard = pu21_metric(ldr_reinhard, ldr_reference, 'SSIM')
    
    print("色调映射方法评估:")
    print(f"  Gamma (2.2):     PSNR = {psnr_gamma:.2f} dB, SSIM = {ssim_gamma:.4f}")
    print(f"  Reinhard:        PSNR = {psnr_reinhard:.2f} dB, SSIM = {ssim_reinhard:.4f}")

def main():
    """
    主函数：运行所有示例
    """
    print("PU21 Python实现演示")
    print("=" * 50)
    
    try:
        # 运行各种示例
        example_pu21_encoding()
        example_different_encodings()
        example_hdr_metrics()
        example_custom_metric()
        example_tone_mapping_evaluation()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 