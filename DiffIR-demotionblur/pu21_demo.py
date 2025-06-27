#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU21编码简化演示脚本
快速展示PU21编码的核心特性
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加metrics目录到路径
sys.path.append('./metrics')
from pu21_encoder import PU21Encoder

def demo_basic_encoding():
    """基本编码演示"""
    print("="*60)
    print("PU21编码基本演示")
    print("="*60)
    
    # 创建编码器
    encoder = PU21Encoder('banding_glare')  # 推荐的编码类型
    
    # 测试常见亮度值
    luminance_examples = {
        0.01: "星光",
        0.1: "月光", 
        1.0: "烛光",
        10.0: "室内照明",
        100.0: "办公室照明",
        1000.0: "阴天户外",
        10000.0: "直射阳光"
    }
    
    print(f"{'场景':<12} {'亮度(cd/m²)':<12} {'编码值':<8} {'解码值':<12} {'误差(%)':<8}")
    print("-" * 65)
    
    for lum_val, description in luminance_examples.items():
        if encoder.L_min <= lum_val <= encoder.L_max:
            encoded = encoder.encode(np.array([lum_val]))[0]
            decoded = encoder.decode(np.array([encoded]))[0]
            error = abs(lum_val - decoded) / lum_val * 100
            
            print(f"{description:<12} {lum_val:<12.2f} {encoded:<8.2f} {decoded:<12.3f} {error:<8.4f}")
        else:
            print(f"{description:<12} {lum_val:<12.2f} {'超范围':<8} {'-':<12} {'-':<8}")

def plot_encoding_comparison():
    """绘制编码对比图"""
    print("\n" + "="*60)
    print("生成PU21编码对比图")
    print("="*60)
    
    # 创建亮度范围
    Y_values = np.logspace(np.log10(0.005), np.log10(10000), 500)
    
    # 不同编码类型
    encoding_types = ['banding', 'banding_glare', 'peaks', 'peaks_glare']
    colors = ['blue', 'red', 'green', 'orange']
    
    plt.figure(figsize=(12, 8))
    
    # 子图1: 编码曲线
    plt.subplot(2, 2, 1)
    for i, enc_type in enumerate(encoding_types):
        encoder = PU21Encoder(enc_type)
        V_encoded = encoder.encode(Y_values)
        plt.semilogx(Y_values, V_encoded, color=colors[i], linewidth=2, 
                    label=enc_type, alpha=0.8)
    
    plt.xlabel('线性亮度值 (cd/m²)')
    plt.ylabel('编码值')
    plt.title('PU21编码曲线对比')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 标记重要点
    important_points = [0.1, 1, 10, 100, 1000]
    for point in important_points:
        plt.axvline(x=point, color='gray', linestyle='--', alpha=0.5)
    
    # 子图2: 推荐编码的详细分析
    plt.subplot(2, 2, 2)
    encoder_recommended = PU21Encoder('banding_glare')
    V_recommended = encoder_recommended.encode(Y_values)
    
    plt.semilogx(Y_values, V_recommended, 'r-', linewidth=3, label='banding_glare (推荐)')
    
    # 标记关键点
    key_points = [0.1, 1, 10, 100, 1000]
    for point in key_points:
        encoded_val = encoder_recommended.encode(np.array([point]))[0]
        plt.scatter(point, encoded_val, color='red', s=100, zorder=5)
        plt.annotate(f'{point} cd/m²\\nV={encoded_val:.1f}', 
                    xy=(point, encoded_val), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.xlabel('线性亮度值 (cd/m²)')
    plt.ylabel('编码值')
    plt.title('推荐编码类型详细分析')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图3: 编码精度测试
    plt.subplot(2, 2, 3)
    test_range = np.logspace(np.log10(0.005), np.log10(10000), 100)
    encoded_test = encoder_recommended.encode(test_range)
    decoded_test = encoder_recommended.decode(encoded_test)
    
    plt.loglog(test_range, decoded_test, 'b-', linewidth=2, label='解码值')
    plt.loglog(test_range, test_range, 'r--', linewidth=1, label='理想值 (y=x)')
    
    plt.xlabel('原始亮度值 (cd/m²)')
    plt.ylabel('解码亮度值 (cd/m²)')
    plt.title('编码解码精度测试')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图4: 编码分布示例
    plt.subplot(2, 2, 4)
    
    # 模拟HDR图像亮度分布
    np.random.seed(42)
    indoor_samples = np.random.lognormal(np.log(1), 1, 2000)
    outdoor_samples = np.random.lognormal(np.log(100), 0.8, 1500)
    bright_samples = np.random.lognormal(np.log(1000), 0.5, 500)
    
    all_samples = np.concatenate([indoor_samples, outdoor_samples, bright_samples])
    all_samples = np.clip(all_samples, 0.005, 10000)
    
    # 原始分布
    plt.hist(all_samples, bins=50, alpha=0.5, color='skyblue', 
             label='原始分布', density=True)
    
    # 编码后分布
    encoded_samples = encoder_recommended.encode(all_samples)
    plt.hist(encoded_samples, bins=50, alpha=0.5, color='lightcoral', 
             label='编码后分布', density=True)
    
    plt.xlabel('值')
    plt.ylabel('密度')
    plt.title('HDR图像亮度分布示例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pu21_demo_results.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'pu21_demo_results.png'")
    plt.show()

def interactive_test():
    """交互式测试"""
    print("\n" + "="*60)
    print("PU21编码交互式测试")
    print("="*60)
    
    encoder = PU21Encoder('banding_glare')
    
    print(f"有效亮度范围: {encoder.L_min} - {encoder.L_max} cd/m²")
    print("输入亮度值进行编码测试，输入 'q' 退出")
    
    while True:
        try:
            user_input = input("\n请输入亮度值 (cd/m²): ")
            if user_input.lower() == 'q':
                break
                
            luminance = float(user_input)
            
            if encoder.L_min <= luminance <= encoder.L_max:
                encoded = encoder.encode(np.array([luminance]))[0]
                decoded = encoder.decode(np.array([encoded]))[0]
                error = abs(luminance - decoded) / luminance * 100
                
                print(f"输入亮度: {luminance:.3f} cd/m²")
                print(f"编码值: {encoded:.2f}")
                print(f"解码值: {decoded:.3f} cd/m²")
                print(f"相对误差: {error:.4f}%")
                
                # 给出亮度级别参考
                if luminance < 0.1:
                    level = "极暗 (星光级别)"
                elif luminance < 1:
                    level = "很暗 (月光级别)"
                elif luminance < 10:
                    level = "暗 (烛光级别)"
                elif luminance < 100:
                    level = "中等 (室内照明)"
                elif luminance < 1000:
                    level = "明亮 (办公室照明)"
                else:
                    level = "非常明亮 (户外阳光)"
                
                print(f"亮度级别: {level}")
                
            else:
                print(f"输入值 {luminance} 超出有效范围 ({encoder.L_min}-{encoder.L_max})")
                
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n退出交互式测试")
            break
    
    print("交互式测试结束")

def show_encoding_info():
    """显示编码信息"""
    print("\n" + "="*60)
    print("PU21编码技术信息")
    print("="*60)
    
    info = """
PU21 (Perceptually Uniform 2021) 编码特性:

1. 设计目标:
   • 感知均匀性: 相等的编码差异对应相等的感知差异
   • HDR适配: 专门为高动态范围内容设计
   • 质量评估: 可用于现有SDR图像质量指标

2. 技术参数:
   • 输入范围: 0.005 - 10,000 cd/m² (绝对亮度单位)
   • 输出范围: 0 - ~600 (编码值)
   • 参考映射: 100 cd/m² → 256 (模拟SDR)

3. 编码类型:
   • banding: 基础条带伪影优化
   • banding_glare: 条带+眩光优化 (推荐)
   • peaks: 峰值保持优化
   • peaks_glare: 峰值+眩光优化

4. 应用场景:
   • HDR图像质量评估
   • HDR内容的感知均匀编码
   • 传统SDR指标的HDR扩展

5. 优势:
   • 基于人眼视觉特性
   • 高精度编码解码
   • 与现有工具兼容
   • 多种优化选项
    """
    
    print(info)

def main():
    """主函数"""
    print("PU21编码演示工具")
    print("作者: HDR图像处理团队")
    print("版本: 1.0")
    
    try:
        # 1. 基本编码演示
        demo_basic_encoding()
        
        # 2. 显示技术信息
        show_encoding_info()
        
        # 3. 生成可视化图表
        plot_encoding_comparison()
        
        # 4. 交互式测试 (可选)
        user_choice = input("\n是否进行交互式测试? (y/n): ")
        if user_choice.lower() == 'y':
            interactive_test()
        
        print("\n" + "="*60)
        print("演示完成!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        print("请检查依赖项和文件路径")

if __name__ == '__main__':
    main() 