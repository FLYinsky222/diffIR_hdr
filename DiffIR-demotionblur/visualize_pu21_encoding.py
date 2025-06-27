#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU21编码可视化脚本
展示PU21编码的特性、不同编码类型的对比以及编码解码测试
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import sys
import os

# 添加metrics目录到路径
sys.path.append('./metrics')
from pu21_encoder import PU21Encoder, get_luminance

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_encoding_curves():
    """绘制不同PU21编码类型的编码曲线"""
    # 创建线性亮度值范围
    Y_linear = np.logspace(np.log10(0.005), np.log10(10000), 1000)
    
    # 不同编码类型
    encoding_types = ['banding', 'banding_glare', 'peaks', 'peaks_glare']
    colors = ['blue', 'red', 'green', 'orange']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制编码曲线
    for i, enc_type in enumerate(encoding_types):
        encoder = PU21Encoder(enc_type)
        V_encoded = encoder.encode(Y_linear)
        
        ax1.semilogx(Y_linear, V_encoded, color=colors[i], linewidth=2, 
                    label=f'{enc_type}', alpha=0.8)
    
    ax1.set_xlabel('线性亮度值 Y (cd/m²)', fontsize=12)
    ax1.set_ylabel('编码值 V', fontsize=12)
    ax1.set_title('PU21编码曲线对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0.005, 10000])
    
    # 添加重要亮度点标记
    important_points = [0.1, 1, 10, 100, 1000]
    for point in important_points:
        ax1.axvline(x=point, color='gray', linestyle='--', alpha=0.5)
        ax1.text(point, ax1.get_ylim()[1]*0.9, f'{point}', 
                rotation=90, ha='right', va='top', fontsize=9)
    
    # 绘制编码导数（敏感度）
    Y_derivative = Y_linear[1:-1]
    for i, enc_type in enumerate(encoding_types):
        encoder = PU21Encoder(enc_type)
        V_encoded = encoder.encode(Y_linear)
        
        # 计算导数 dV/dY
        dV_dY = np.diff(V_encoded) / np.diff(Y_linear)
        
        ax2.loglog(Y_derivative, dV_dY, color=colors[i], linewidth=2, 
                  label=f'{enc_type}', alpha=0.8)
    
    ax2.set_xlabel('线性亮度值 Y (cd/m²)', fontsize=12)
    ax2.set_ylabel('编码敏感度 dV/dY', fontsize=12)
    ax2.set_title('PU21编码敏感度对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_encoding_decoding_test():
    """测试编码解码的精度"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    encoding_types = ['banding', 'banding_glare', 'peaks', 'peaks_glare']
    
    for i, enc_type in enumerate(encoding_types):
        encoder = PU21Encoder(enc_type)
        
        # 测试值
        Y_test = np.logspace(np.log10(0.005), np.log10(10000), 100)
        V_encoded = encoder.encode(Y_test)
        Y_decoded = encoder.decode(V_encoded)
        
        # 计算相对误差
        relative_error = np.abs(Y_test - Y_decoded) / Y_test * 100
        
        # 绘制原始值 vs 解码值
        axes[i].loglog(Y_test, Y_decoded, 'b-', linewidth=2, label='解码值')
        axes[i].loglog(Y_test, Y_test, 'r--', linewidth=1, label='理想值 (y=x)')
        
        axes[i].set_xlabel('原始亮度值 (cd/m²)')
        axes[i].set_ylabel('解码亮度值 (cd/m²)')
        axes[i].set_title(f'{enc_type}\n最大相对误差: {np.max(relative_error):.3f}%')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # 添加误差信息
        mean_error = np.mean(relative_error)
        axes[i].text(0.05, 0.95, f'平均误差: {mean_error:.3f}%', 
                    transform=axes[i].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.suptitle('PU21编码解码精度测试', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_hdr_range_visualization():
    """可视化HDR亮度范围和编码映射"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1])
    
    # 主编码曲线图
    ax_main = fig.add_subplot(gs[:, 0])
    
    # 使用推荐的banding_glare编码
    encoder = PU21Encoder('banding_glare')
    Y_range = np.logspace(np.log10(0.005), np.log10(10000), 1000)
    V_encoded = encoder.encode(Y_range)
    
    ax_main.semilogx(Y_range, V_encoded, 'b-', linewidth=3, label='PU21编码曲线')
    
    # 标记重要的亮度级别
    luminance_levels = {
        0.01: '星光',
        0.1: '月光',
        1: '烛光',
        10: '室内照明',
        100: '办公室照明',
        1000: '阴天户外',
        10000: '直射阳光'
    }
    
    colors_levels = plt.cm.viridis(np.linspace(0, 1, len(luminance_levels)))
    
    for i, (lum, desc) in enumerate(luminance_levels.items()):
        encoded_val = encoder.encode(np.array([lum]))[0]
        ax_main.scatter(lum, encoded_val, color=colors_levels[i], s=100, zorder=5)
        ax_main.annotate(f'{desc}\n{lum} cd/m²\nV={encoded_val:.1f}', 
                        xy=(lum, encoded_val), xytext=(10, 10),
                        textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors_levels[i], alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    ax_main.set_xlabel('线性亮度值 (cd/m²)', fontsize=12)
    ax_main.set_ylabel('PU21编码值', fontsize=12)
    ax_main.set_title('HDR亮度范围与PU21编码映射', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    
    # 亮度分布直方图
    ax_hist = fig.add_subplot(gs[0, 1])
    
    # 模拟典型HDR图像的亮度分布
    np.random.seed(42)
    # 多峰分布模拟室内外混合场景
    samples1 = np.random.lognormal(np.log(1), 1, 3000)    # 室内
    samples2 = np.random.lognormal(np.log(100), 0.5, 2000)  # 户外阴影
    samples3 = np.random.lognormal(np.log(1000), 0.3, 1000) # 户外明亮
    
    all_samples = np.concatenate([samples1, samples2, samples3])
    all_samples = np.clip(all_samples, 0.005, 10000)
    
    ax_hist.hist(all_samples, bins=50, alpha=0.7, color='skyblue', 
                orientation='horizontal', density=True)
    ax_hist.set_ylim([0.005, 10000])
    ax_hist.set_yscale('log')
    ax_hist.set_xlabel('密度')
    ax_hist.set_title('典型HDR图像\n亮度分布', fontsize=12)
    ax_hist.grid(True, alpha=0.3)
    
    # 编码值分布
    ax_encoded = fig.add_subplot(gs[1, 1])
    encoded_samples = encoder.encode(all_samples)
    ax_encoded.hist(encoded_samples, bins=50, alpha=0.7, color='lightcoral', 
                   orientation='horizontal', density=True)
    ax_encoded.set_xlabel('密度')
    ax_encoded.set_ylabel('编码值')
    ax_encoded.set_title('对应的PU21\n编码分布', fontsize=12)
    ax_encoded.grid(True, alpha=0.3)
    
    # 编码效率信息
    ax_info = fig.add_subplot(gs[2, 1])
    ax_info.axis('off')
    
    info_text = f"""
PU21编码特性:
• 输入范围: 0.005 - 10,000 cd/m²
• 输出范围: 0 - ~600
• 100 cd/m² → 256 (模拟SDR)
• 感知均匀性优化
• 适用于HDR图像质量评估

编码类型说明:
• banding: 基础条带优化
• banding_glare: 条带+眩光优化 (推荐)
• peaks: 峰值优化  
• peaks_glare: 峰值+眩光优化
    """
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig

def compare_with_traditional_encoding():
    """对比PU21与传统编码方法"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    Y_range = np.logspace(np.log10(0.005), np.log10(10000), 1000)
    
    # PU21编码
    pu21_encoder = PU21Encoder('banding_glare')
    V_pu21 = pu21_encoder.encode(Y_range)
    
    # 传统编码方法
    # 1. 对数编码
    V_log = np.log10(Y_range + 1e-6) * 100 + 300  # 归一化到合理范围
    
    # 2. 伽马编码 (类似sRGB)
    V_gamma = np.power(Y_range / 10000, 1/2.2) * 600
    
    # 3. 线性编码
    V_linear = (Y_range - 0.005) / (10000 - 0.005) * 600
    
    # 绘制编码曲线对比
    axes[0, 0].semilogx(Y_range, V_pu21, 'r-', linewidth=2, label='PU21')
    axes[0, 0].semilogx(Y_range, V_log, 'g--', linewidth=2, label='对数编码')
    axes[0, 0].semilogx(Y_range, V_gamma, 'b:', linewidth=2, label='伽马编码')
    axes[0, 0].semilogx(Y_range, V_linear, 'm-.', linewidth=2, label='线性编码')
    
    axes[0, 0].set_xlabel('线性亮度值 (cd/m²)')
    axes[0, 0].set_ylabel('编码值')
    axes[0, 0].set_title('编码方法对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 计算并绘制编码敏感度
    Y_derivative = Y_range[1:-1]
    
    dV_dY_pu21 = np.diff(V_pu21) / np.diff(Y_range)
    dV_dY_log = np.diff(V_log) / np.diff(Y_range)
    dV_dY_gamma = np.diff(V_gamma) / np.diff(Y_range)
    dV_dY_linear = np.diff(V_linear) / np.diff(Y_range)
    
    axes[0, 1].loglog(Y_derivative, dV_dY_pu21, 'r-', linewidth=2, label='PU21')
    axes[0, 1].loglog(Y_derivative, dV_dY_log, 'g--', linewidth=2, label='对数编码')
    axes[0, 1].loglog(Y_derivative, dV_dY_gamma, 'b:', linewidth=2, label='伽马编码')
    axes[0, 1].loglog(Y_derivative, dV_dY_linear, 'm-.', linewidth=2, label='线性编码')
    
    axes[0, 1].set_xlabel('线性亮度值 (cd/m²)')
    axes[0, 1].set_ylabel('编码敏感度 dV/dY')
    axes[0, 1].set_title('编码敏感度对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 模拟感知差异
    # 创建两个相近的亮度值
    Y1 = np.array([0.1, 1, 10, 100, 1000])
    Y2 = Y1 * 1.1  # 10%的差异
    
    # 计算不同编码下的差异
    diff_pu21 = np.abs(pu21_encoder.encode(Y2) - pu21_encoder.encode(Y1))
    diff_log = np.abs(np.log10(Y2 + 1e-6) * 100 + 300 - (np.log10(Y1 + 1e-6) * 100 + 300))
    diff_gamma = np.abs(np.power(Y2 / 10000, 1/2.2) * 600 - np.power(Y1 / 10000, 1/2.2) * 600)
    diff_linear = np.abs((Y2 - 0.005) / (10000 - 0.005) * 600 - (Y1 - 0.005) / (10000 - 0.005) * 600)
    
    x_pos = np.arange(len(Y1))
    width = 0.2
    
    axes[1, 0].bar(x_pos - 1.5*width, diff_pu21, width, label='PU21', color='red', alpha=0.7)
    axes[1, 0].bar(x_pos - 0.5*width, diff_log, width, label='对数编码', color='green', alpha=0.7)
    axes[1, 0].bar(x_pos + 0.5*width, diff_gamma, width, label='伽马编码', color='blue', alpha=0.7)
    axes[1, 0].bar(x_pos + 1.5*width, diff_linear, width, label='线性编码', color='magenta', alpha=0.7)
    
    axes[1, 0].set_xlabel('基础亮度值 (cd/m²)')
    axes[1, 0].set_ylabel('编码差异值')
    axes[1, 0].set_title('10%亮度差异的编码响应')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'{y:.1f}' for y in Y1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 优势总结
    axes[1, 1].axis('off')
    summary_text = """
PU21编码优势:

1. 感知均匀性
   • 基于人眼视觉特性设计
   • 相等的编码差异对应相等的感知差异

2. HDR适配性
   • 专门为HDR内容优化
   • 覆盖完整的HDR亮度范围

3. 质量评估友好
   • 可直接用于现有SDR质量指标
   • 100 cd/m² 映射到256 (SDR参考)

4. 多种优化选项
   • banding: 条带伪影优化
   • glare: 眩光效应优化
   • peaks: 峰值保持优化

5. 编码效率
   • 在感知重要区域分配更多编码位
   • 提高编码资源利用效率
    """
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_interactive_demo():
    """创建交互式演示"""
    print("\n" + "="*60)
    print("PU21编码交互式演示")
    print("="*60)
    
    # 选择编码类型
    print("\n可用的编码类型:")
    encoding_types = ['banding', 'banding_glare', 'peaks', 'peaks_glare']
    for i, enc_type in enumerate(encoding_types):
        print(f"{i+1}. {enc_type}")
    
    while True:
        try:
            choice = int(input("\n请选择编码类型 (1-4): ")) - 1
            if 0 <= choice < len(encoding_types):
                selected_type = encoding_types[choice]
                break
            else:
                print("请输入1-4之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    encoder = PU21Encoder(selected_type)
    print(f"\n已选择: {selected_type}")
    
    # 测试编码
    print("\n" + "-"*40)
    print("亮度值编码测试")
    print("-"*40)
    
    test_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    print(f"{'亮度值 (cd/m²)':<15} {'编码值':<10} {'解码值':<15} {'误差 (%)':<10}")
    print("-" * 55)
    
    for y_val in test_values:
        if encoder.L_min <= y_val <= encoder.L_max:
            v_encoded = encoder.encode(np.array([y_val]))[0]
            y_decoded = encoder.decode(np.array([v_encoded]))[0]
            error = abs(y_val - y_decoded) / y_val * 100
            
            print(f"{y_val:<15.3f} {v_encoded:<10.2f} {y_decoded:<15.3f} {error:<10.4f}")
        else:
            print(f"{y_val:<15.3f} {'超出范围':<10} {'-':<15} {'-':<10}")
    
    # 自定义输入测试
    print("\n" + "-"*40)
    print("自定义亮度值测试")
    print("-"*40)
    
    while True:
        try:
            user_input = input(f"\n请输入亮度值 ({encoder.L_min}-{encoder.L_max} cd/m²，输入'q'退出): ")
            if user_input.lower() == 'q':
                break
                
            y_val = float(user_input)
            if encoder.L_min <= y_val <= encoder.L_max:
                v_encoded = encoder.encode(np.array([y_val]))[0]
                y_decoded = encoder.decode(np.array([v_encoded]))[0]
                error = abs(y_val - y_decoded) / y_val * 100
                
                print(f"原始值: {y_val:.3f} cd/m²")
                print(f"编码值: {v_encoded:.2f}")
                print(f"解码值: {y_decoded:.3f} cd/m²")
                print(f"相对误差: {error:.4f}%")
            else:
                print(f"输入值超出有效范围 ({encoder.L_min}-{encoder.L_max})")
                
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            break
    
    print("\n演示结束!")

def main():
    """主函数"""
    print("PU21编码可视化工具")
    print("="*50)
    
    # 创建输出目录
    output_dir = "pu21_visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成可视化图表...")
    
    # 1. 基本编码曲线
    print("1. 生成编码曲线对比图...")
    fig1 = plot_encoding_curves()
    fig1.savefig(f'{output_dir}/pu21_encoding_curves.png', dpi=300, bbox_inches='tight')
    
    # 2. 编码解码精度测试
    print("2. 生成编码解码精度测试图...")
    fig2 = plot_encoding_decoding_test()
    fig2.savefig(f'{output_dir}/pu21_accuracy_test.png', dpi=300, bbox_inches='tight')
    
    # 3. HDR范围可视化
    print("3. 生成HDR范围可视化图...")
    fig3 = plot_hdr_range_visualization()
    fig3.savefig(f'{output_dir}/pu21_hdr_range.png', dpi=300, bbox_inches='tight')
    
    # 4. 与传统编码方法对比
    print("4. 生成编码方法对比图...")
    fig4 = compare_with_traditional_encoding()
    fig4.savefig(f'{output_dir}/pu21_comparison.png', dpi=300, bbox_inches='tight')
    
    print(f"\n所有图表已保存到 '{output_dir}' 目录")
    
    # 显示图表
    plt.show()
    
    # 交互式演示
    try:
        create_interactive_demo()
    except KeyboardInterrupt:
        print("\n\n程序已退出")

if __name__ == '__main__':
    main() 