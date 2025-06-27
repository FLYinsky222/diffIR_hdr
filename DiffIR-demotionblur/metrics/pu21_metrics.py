#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from metrics.pu21_encoder import PU21Encoder, get_luminance, is_image
import warnings

def pu21_metric(I_test, I_reference, metric='PSNR', display_model=None, crf_correction=False):
    """
    A convenience function for calling traditional (SDR) metrics on
    PU-encoded pixel values. This is useful for adapting traditional metrics
    to HDR images.
    
    Args:
        I_test (np.ndarray): Test image
        I_reference (np.ndarray): Reference image  
        metric (str or callable): Metric to compute. Options:
            - 'PSNR': Peak Signal-to-Noise Ratio
            - 'PSNRY': Peak Signal-to-Noise Ratio on luminance
            - 'SSIM': Structural Similarity Index (on luminance)
            - 'MSSSIM': Multi-Scale Structural Similarity Index (on luminance)
            - Or a callable function Q = fun(I_test, I_reference)
        display_model: Display model object (not implemented yet)
        crf_correction (bool): Apply CRF correction for SI-HDR evaluation
    
    Returns:
        float: Quality metric value
    
    Notes:
        When no display model is passed, I_test and I_reference must be provided
        as ABSOLUTE linear colour or luminance values.
        
        When display model is passed, I_test and I_reference contain images in
        display-encoded sRGB colour space (not implemented yet).
    """
    
    # Validate inputs
    if not is_image(I_test) or not is_image(I_reference):
        raise ValueError("Inputs must be valid images")
    
    if I_test.shape != I_reference.shape:
        raise ValueError("Test and reference images must have the same shape")
    
    # Convert integer images to float
    if not np.issubdtype(I_test.dtype, np.floating):
        if np.issubdtype(I_test.dtype, np.integer):
            max_val = np.iinfo(I_test.dtype).max
            I_test = I_test.astype(np.float32) / max_val
        else:
            I_test = I_test.astype(np.float32)
    
    if not np.issubdtype(I_reference.dtype, np.floating):
        if np.issubdtype(I_reference.dtype, np.integer):
            max_val = np.iinfo(I_reference.dtype).max
            I_reference = I_reference.astype(np.float32) / max_val
        else:
            I_reference = I_reference.astype(np.float32)
    
    if display_model is not None:
        # Simulate an SDR display if display model is provided
        # TODO: Implement display model
        raise NotImplementedError("Display model not implemented yet")
    else:
        # If no display model is provided, we assume an HDR image in absolute units
        L_test = I_test
        L_reference = I_reference
    
    if crf_correction:
        if L_test.ndim != 3:
            raise ValueError('crf_correction can be used with color images only.')
        # TODO: Implement CRF correction
        raise NotImplementedError("CRF correction not implemented yet")
    
    # Define metric properties
    metric_props = {
        'PSNR': {'only_lum': False, 'func': lambda T, R: psnr_pu21(T, R)},
        'PSNRY': {'only_lum': True, 'func': lambda T, R: psnr_pu21(T, R)},
        'SSIM': {'only_lum': True, 'func': lambda T, R: ssim_pu21(T, R)},
        'MSSSIM': {'only_lum': True, 'func': lambda T, R: msssim_pu21(T, R)},
    }
    
    # Create PU21 encoder
    pu21 = PU21Encoder()
    
    if isinstance(metric, str):
        metric = metric.upper()
        if metric in metric_props:
            # Convert RGB image to luminance if needed
            if L_test.ndim == 3 and metric_props[metric]['only_lum']:
                L_test = get_luminance(L_test)
                L_reference = get_luminance(L_reference)
            
            # Convert absolute linear values to PU values
            P_test = pu21.encode(L_test)
            P_reference = pu21.encode(L_reference)
            
            # Compute metric
            Q = metric_props[metric]['func'](P_test, P_reference)
        else:
            raise ValueError(f'Unknown metric "{metric}"')
    else:
        # Custom metric function
        if not callable(metric):
            raise ValueError("Metric must be a string or callable function")
        
        # Convert absolute linear values to PU values
        P_test = pu21.encode(L_test)
        P_reference = pu21.encode(L_reference)
        Q = metric(P_test, P_reference)
    
    return Q


def psnr_pu21(I_test, I_reference, data_range=530):
    """
    Compute PSNR for PU21-encoded images
    
    Args:
        I_test (np.ndarray): Test image
        I_reference (np.ndarray): Reference image
        data_range (float): Data range for PSNR calculation
    
    Returns:
        float: PSNR value in dB
    """
    return peak_signal_noise_ratio(I_reference, I_test, data_range=data_range)


def ssim_pu21(I_test, I_reference, data_range=600):
    """
    Compute SSIM for PU21-encoded images
    
    Args:
        I_test (np.ndarray): Test image
        I_reference (np.ndarray): Reference image
        data_range (float): Data range for SSIM calculation
    
    Returns:
        float: SSIM value
    """
    return structural_similarity(I_reference, I_test, data_range=data_range)


def msssim_pu21(I_test, I_reference, data_range=256):
    """
    Compute Multi-Scale SSIM for PU21-encoded images
    
    Args:
        I_test (np.ndarray): Test image
        I_reference (np.ndarray): Reference image
        data_range (float): Data range for MS-SSIM calculation
    
    Returns:
        float: MS-SSIM value
    """
    # Simple implementation using multiple scales
    # For a more sophisticated implementation, consider using other libraries
    scales = [1, 2, 4]
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # Standard MS-SSIM weights
    
    if len(scales) > len(weights):
        scales = scales[:len(weights)]
    
    ssim_values = []
    
    for i, scale in enumerate(scales):
        if scale == 1:
            # Original scale
            ssim_val = structural_similarity(I_reference, I_test, data_range=data_range)
        else:
            # Downsample
            from skimage.transform import rescale
            factor = 1.0 / scale
            I_test_scaled = rescale(I_test, factor, anti_aliasing=True, channel_axis=None if I_test.ndim == 2 else -1)
            I_ref_scaled = rescale(I_reference, factor, anti_aliasing=True, channel_axis=None if I_reference.ndim == 2 else -1)
            ssim_val = structural_similarity(I_ref_scaled, I_test_scaled, data_range=data_range)
        
        ssim_values.append(ssim_val)
    
    # Weighted average
    ms_ssim = np.prod([val**weights[i] for i, val in enumerate(ssim_values)])
    return ms_ssim


def pu21_encode_image(image, L_peak=4000):
    """
    Convenience function to encode an HDR image using PU21
    
    Args:
        image (np.ndarray): HDR image in relative units
        L_peak (float): Peak luminance of HDR display in cd/m^2
    
    Returns:
        np.ndarray: PU21-encoded image
    """
    # Map peak value in image to peak value of display
    image_abs = image / np.max(image) * L_peak
    
    # Create encoder and encode
    pu21 = PU21Encoder()
    encoded = pu21.encode(image_abs)
    
    return encoded


if __name__ == '__main__':
    # 简单测试
    import numpy as np
    
    # 创建测试图像
    np.random.seed(42)
    
    # 模拟HDR图像（绝对亮度单位）
    I_ref = np.random.rand(64, 64, 3) * 1000  # 0-1000 cd/m^2
    
    # 添加噪声
    I_test_noise = np.maximum(I_ref + I_ref * np.random.randn(*I_ref.shape) * 0.2, 0.05)
    
    # 添加模糊
    from scipy.ndimage import gaussian_filter
    I_test_blur = gaussian_filter(I_ref, sigma=1.0)
    
    # 计算metrics
    psnr_noise = pu21_metric(I_test_noise, I_ref, 'PSNR')
    ssim_noise = pu21_metric(I_test_noise, I_ref, 'SSIM')
    
    psnr_blur = pu21_metric(I_test_blur, I_ref, 'PSNR')
    ssim_blur = pu21_metric(I_test_blur, I_ref, 'SSIM')
    
    print(f'噪声图像: PSNR = {psnr_noise:.2f} dB, SSIM = {ssim_noise:.4f}')
    print(f'模糊图像: PSNR = {psnr_blur:.2f} dB, SSIM = {ssim_blur:.4f}') 