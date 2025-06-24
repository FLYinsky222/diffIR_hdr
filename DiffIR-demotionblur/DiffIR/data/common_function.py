import os
import numpy as np
import cv2

def extract_info_from_dgain_prompt(prompt_content):
    """
    从dgain_prompt内容中提取信息
    
    Args:
        prompt_content: prompt文件内容
        
    Returns:
        dict: 包含所有字段信息的字典
    """
    info = {
        'dgain': 1.0,
        'gamma': 1.6,
        'strategy': 'reconstruct noise-degraded structures and apply luminance mapping by sksldrlogtohdrlog',
        'PSNR': 24.36,
        'PU_PSNR': 155.28,
        'overexposure_constraint_satisfied': True,
        'original_overexposed_pixels': 11996,
        'predicted_overexposed_pixels': 1520,
        'overexposure_ratio': 0.127
    }
    
    lines = prompt_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        
        # 解析dgain
        if line.startswith('dgain:'):
            try:
                info['dgain'] = float(line.split(':', 1)[1].strip())
            except:
                pass
        
        # 解析gamma
        elif line.startswith('gamma:'):
            try:
                info['gamma'] = float(line.split(':', 1)[1].strip())
            except:
                pass
        
        # 解析strategy
        elif line.startswith('strategy:'):
            strategy_value = line.split(':', 1)[1].strip()
            # 确保strategy中包含"and apply luminance mapping"
            if 'by sksldrlogtohdrlog' in strategy_value and 'and apply luminance mapping' not in strategy_value:
                strategy_value = strategy_value.replace('by sksldrlogtohdrlog', 'and apply luminance mapping by sksldrlogtohdrlog')
            info['strategy'] = strategy_value
        
        # 解析PSNR
        elif line.startswith('PSNR:'):
            try:
                info['PSNR'] = float(line.split(':', 1)[1].strip())
            except:
                pass
        
        # 解析PU_PSNR
        elif line.startswith('PU_PSNR:'):
            try:
                info['PU_PSNR'] = float(line.split(':', 1)[1].strip())
            except:
                pass
        
        # 解析overexposure_constraint_satisfied
        elif line.startswith('overexposure_constraint_satisfied:'):
            try:
                value = line.split(':', 1)[1].strip().lower()
                info['overexposure_constraint_satisfied'] = value == 'true'
            except:
                pass
        
        # 解析original_overexposed_pixels
        elif line.startswith('original_overexposed_pixels:'):
            try:
                info['original_overexposed_pixels'] = int(line.split(':', 1)[1].strip())
            except:
                pass
        
        # 解析predicted_overexposed_pixels
        elif line.startswith('predicted_overexposed_pixels:'):
            try:
                info['predicted_overexposed_pixels'] = int(line.split(':', 1)[1].strip())
            except:
                pass
        
        # 解析overexposure_ratio
        elif line.startswith('overexposure_ratio:'):
            try:
                info['overexposure_ratio'] = float(line.split(':', 1)[1].strip())
            except:
                pass
    
    return info

def log_tone_mapping(hdr,hdr_max,epsilon=1e-6):
    return np.log(1 + hdr) / np.log(1 + hdr_max + epsilon)

def inverse_log_tone_mapping(ldr, hdr_max, epsilon=1e-6):
    """
    ldr: 经过 log_tone_mapping 后的图像（范围应为 0~1）
    hdr_max: 映射前的最大值（必须与 log_tone_mapping 中的一致）
    """
    return np.exp(ldr * np.log(1 + hdr_max + epsilon)) - 1


def load_ldr_file(file_path):
    """
    加载LDR文件
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件：{file_path}")
    
    try:
        # 读取LDR图像
        ldr_image = cv2.imread(file_path)
        if ldr_image is None:
            raise ValueError(f"无法读取LDR文件：{file_path}")
        
        # 转换BGR到RGB，与hdr_to_png.py保持一致
        #ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB)
        
        # 转换为浮点型并归一化到0-1
        ldr_image = ldr_image.astype(np.float32) / 255.0
        
        return ldr_image
        
    except Exception as e:
        raise Exception(f"读取LDR文件时出错：{str(e)}")

def load_hdr(file_path):
    """
    加载HDR图像文件
    
    Args:
        file_path: HDR文件路径
        
    Returns:
        加载的HDR图像数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件：{file_path}")
    
    try:
        # 读取HDR图像
        hdr_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        #hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
        if hdr_image is None:
            raise ValueError(f"无法读取HDR文件：{file_path}")
            
        # 确保图像是3通道的
        if len(hdr_image.shape) == 2:  # 如果是单通道图像
            hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_GRAY2BGR)
            
        # 确保图像是浮点型
        if hdr_image.dtype != np.float32:
            hdr_image = hdr_image.astype(np.float32)
            
        # 检查图像的通道数和类型
        if len(hdr_image.shape) != 3 or hdr_image.shape[2] != 3:
            raise ValueError(f"图像必须是3通道的：当前shape为{hdr_image.shape}")
            
        #print(f"图像信息：shape={hdr_image.shape}, dtype={hdr_image.dtype}")
        return hdr_image
        
    except Exception as e:
        raise Exception(f"读取HDR文件时出错：{str(e)}")


def custom_tone_mapping(hdr, dgain=1.0, gamma=2.0,max_value=1000):
    """
    自定义色调映射：归一化到0-1，乘以dgain，clip到0-1，再apply gamma
    """
    # 归一化到0-1
    norm = hdr / max_value
    # 乘以dgain
    norm = norm * dgain
    # clip到0-1
    norm = np.clip(norm, 0, 1)
    # gamma校正
    ldr = np.power(norm, 1.0/gamma)
    return ldr

def inverse_custom_tone_mapping(ldr, dgain=1.0, gamma=2.0, max_value=1000.0):
        """
        custom_tone_mapping的反函数：将LDR恢复为HDR
        保持过曝区域的过曝状态
        
        Args:
            ldr: LDR图像 (0-1范围)
            dgain: dgain值，从caption中提取
            gamma: gamma值，默认2.0
            max_value: HDR最大值，默认1000.0
        """
        # 1. 逆gamma校正
        norm_after_clip = np.power(ldr, gamma)
        
        # 2. 识别过曝区域：LDR中接近1.0的区域认为是过曝
        #overexposed_mask = ldr >= 0.95  # 阈值可以调整
        
        # 3. 逆dgain操作
        norm_before_clip = norm_after_clip / dgain
        
        # 4. 对于过曝区域，确保其在HDR中也保持过曝状态
        # 过曝区域应该至少为1.0（即在原始HDR中至少为hdr_max）
        #norm_before_clip = np.where(overexposed_mask, 
        #                           np.maximum(norm_before_clip, 1.2),  # 确保过曝区域超过1
        #                           norm_before_clip)
        norm_after_clip = np.clip(norm_before_clip, 0, 1)
        
        # 5. 恢复到HDR范围 (0-hdr_max)
        hdr = norm_after_clip * max_value
        
        return hdr