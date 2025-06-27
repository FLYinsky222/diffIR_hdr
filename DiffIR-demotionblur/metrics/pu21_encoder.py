#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import warnings

class PU21Encoder:
    """
    Transform absolute linear luminance values to/from the perceptually
    uniform (PU) space. This class is intended for adapting image quality 
    metrics to operate on HDR content.
    
    The derivation of the PU21 encoding is explained in the paper: 
    
    R. Mantiuk and M. Azimi
    PU21: A novel perceptually uniform encoding for adapting existing
    quality metrics for HDR.
    Picture Coding Symposium 2021
    
    The new PU21 encoding improves on the older PU (or PU08) encoding.
    """
    
    def __init__(self, encoding_type='banding_glare'):
        """
        Create PU21 encoder for a given type.
        
        Args:
            encoding_type (str): Type of encoding. Options:
                - 'banding' 
                - 'banding_glare' (recommended, default)
                - 'peaks'
                - 'peaks_glare'
        """
        self.L_min = 0.005  # The minimum linear value (luminance or radiance)
        self.L_max = 10000  # The maximum linear value (luminance or radiance)
        
        # The parameters were updated on 06/02/2020
        if encoding_type == 'banding':
            self.par = np.array([1.070275272, 0.4088273932, 0.153224308, 
                                0.2520326168, 1.063512885, 1.14115047, 521.4527484])
        elif encoding_type == 'banding_glare':
            self.par = np.array([0.353487901, 0.3734658629, 8.277049286e-05, 
                                0.9062562627, 0.09150303166, 0.9099517204, 596.3148142])
        elif encoding_type == 'peaks':
            self.par = np.array([1.043882782, 0.6459495343, 0.3194584211, 
                                0.374025247, 1.114783422, 1.095360363, 384.9217577])
        elif encoding_type == 'peaks_glare':
            self.par = np.array([816.885024, 1479.463946, 0.001253215609, 
                                0.9329636822, 0.06746643971, 1.573435413, 419.6006374])
        else:
            raise ValueError(f'Unknown encoding type: {encoding_type}')
    
    def encode(self, Y):
        """
        Convert from linear (optical) values Y to encoded (electronic) values V
        
        Args:
            Y (np.ndarray): Linear values in the range from 0.005 to 10000. 
                           The values MUST be scaled in absolute units (nits, cd/m^2).
        
        Returns:
            np.ndarray: Encoded values V in the range from 0 to circa 600 
                       (depends on the encoding used). 100 [nit] is mapped to 256 
                       to mimic the input to SDR quality metrics.
        """
        epsilon = 1e-5
        Y = np.asarray(Y, dtype=np.float64)
        
        if np.any(Y < (self.L_min - epsilon)) or np.any(Y > (self.L_max + epsilon)):
            warnings.warn('Values passed to encode are outside the valid range')
        
        # Clamp the values
        Y = np.clip(Y, self.L_min, self.L_max)
        
        p = self.par
        V = np.maximum(
            p[6] * (((p[0] + p[1] * Y**p[3]) / (1 + p[2] * Y**p[3]))**p[4] - p[5]), 
            0
        )
        
        return V
    
    def decode(self, V):
        """
        Convert from encoded (electronic) values V into linear (optical) values Y
        
        Args:
            V (np.ndarray): Encoded values in the range from 0 to circa 600.
        
        Returns:
            np.ndarray: Linear values Y in the range from 0.005 to 10000
        """
        V = np.asarray(V, dtype=np.float64)
        p = self.par
        
        V_p = np.maximum(V / p[6] + p[5], 0)**(1 / p[4])
        Y = (np.maximum(V_p - p[0], 0) / (p[1] - p[2] * V_p))**(1 / p[3])
        
        return Y


def get_luminance(img):
    """
    Return 2D matrix of luminance values for 3D matrix with an RGB image
    
    Args:
        img (np.ndarray): RGB image with shape (H, W, 3)
    
    Returns:
        np.ndarray: Luminance values with shape (H, W)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (H, W, 3)")
    
    # ITU-R BT.709 luminance weights
    Y = img[:, :, 0] * 0.212656 + img[:, :, 1] * 0.715158 + img[:, :, 2] * 0.072186
    return Y


def is_image(I):
    """
    Check if input is a valid image array
    
    Args:
        I: Input to check
    
    Returns:
        bool: True if input is a valid image
    """
    return (isinstance(I, np.ndarray) and 
            (I.ndim == 2 or (I.ndim == 3 and I.shape[2] == 3)))


if __name__ == '__main__':
    # 简单测试
    pu21 = PU21Encoder()
    
    # 测试编码和解码
    Y_test = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
    V_encoded = pu21.encode(Y_test)
    Y_decoded = pu21.decode(V_encoded)
    
    print("原始值:", Y_test)
    print("编码值:", V_encoded)
    print("解码值:", Y_decoded)
    print("误差:", np.abs(Y_test - Y_decoded)) 