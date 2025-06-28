from torch.utils import data as data
from torchvision.transforms.functional import normalize

from DiffIR.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    triple_paths_from_folder,
                                    quadruple_paths_from_folder,
                                    quadruple_paths_from_lmdb)
from DiffIR.data.transforms import (augment, paired_random_crop, paired_random_crop_DP, random_augmentation,
                                    padding_triple, paired_random_crop_triple, random_augmentation_triple)
from DiffIR.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from basicsr.utils.registry import DATASET_REGISTRY
import random
import numpy as np
import torch
import pickle
import cv2
from DiffIR.data.common_function import extract_info_from_dgain_prompt, log_tone_mapping, inverse_log_tone_mapping, load_ldr_file, load_hdr, custom_tone_mapping, inverse_custom_tone_mapping

@DATASET_REGISTRY.register()
class LDR_HDR_HDR_RECOVER_PairedDataset(data.Dataset):
    """Paired image dataset for HDR recovery with gt_recover input.

    Read LQ (Low Quality, LDR image), GT_RECOVER (HDR recovered by other networks) and
    GT (ground truth HDR) image sets.

    The network takes LQ and GT_RECOVER as inputs, and outputs compared with GT for loss calculation.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for ground truth HDR.
            dataroot_lq (str): Data root path for LDR images.
            dataroot_gt_recover (str): Data root path for recovered HDR from other networks.
            dataroot_dgain (str): Data root path for dgain info files.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(LDR_HDR_HDR_RECOVER_PairedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        # 增加gt_recover_folder数据源
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq'] 
        self.gt_recover_folder = opt['dataroot_gt_recover']
        self.dgain_folder = opt['dataroot_dgain']
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            # LMDB模式：支持四个数据源的LMDB数据库
            self.io_backend_opt['db_paths'] = [
                self.lq_folder, 
                self.gt_folder, 
                self.gt_recover_folder, 
                self.dgain_folder
            ]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'gt_recover', 'dgain']
            
            # 使用quadruple_paths_from_lmdb获取四个数据源的路径信息
            self.paths = quadruple_paths_from_lmdb(
                [self.lq_folder, self.gt_folder, self.gt_recover_folder, self.dgain_folder], 
                ['lq', 'gt', 'gt_recover', 'dgain'])
            
            # 为LMDB模式标记
            self.is_lmdb = True
            
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            # meta_info模式：使用元信息文件
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
            self.is_lmdb = False
            
        else:
            # folder模式：使用新的quadruple_paths_from_folder处理四个数据源
            self.paths = quadruple_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.gt_recover_folder, self.dgain_folder], 
                ['lq', 'gt', 'gt_recover', 'dgain'],
                self.filename_tmpl)
            self.is_lmdb = False

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        
        if self.is_lmdb:
            # LMDB模式的数据加载
            lq_path = self.paths[index]['lq_path']
            gt_path = self.paths[index]['gt_path']
            gt_recover_path = self.paths[index]['gt_recover_path']
            dgain_path = self.paths[index]['dgain_path']
            
            # 从LMDB中读取数据
            # LQ图像：使用标准图像编码存储，用imfrombytes读取
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            
            # HDR图像：使用pickle序列化存储，用pickle.loads读取
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = pickle.loads(img_bytes)
            
            # HDR_RECOVER图像：使用pickle序列化存储，用pickle.loads读取
            img_bytes = self.file_client.get(gt_recover_path, 'gt_recover')
            img_gt_recover = pickle.loads(img_bytes)
            
            # 读取dgain信息（以文本形式存储在LMDB中）
            dgain_bytes = self.file_client.get(dgain_path, 'dgain')
            dgain_info = pickle.loads(dgain_bytes)

            dgain = dgain_info['dgain']
            gamma = dgain_info['gamma']
            
            # 设置路径信息用于错误报告
            gt_recover_path = f"lmdb:{gt_recover_path}"
            
        else:
            # Disk模式的数据加载
            # 加载GT HDR图像
            gt_path = self.paths[index]['gt_path']
            try:
                img_gt = load_hdr(gt_path)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            # 加载LQ LDR图像
            lq_path = self.paths[index]['lq_path']
            try:
                img_lq = load_ldr_file(lq_path)
            except:
                raise Exception("lq path {} not working".format(lq_path))

            # 加载GT_RECOVER HDR图像（由其他网络生成）
            gt_recover_path = self.paths[index]['gt_recover_path']
            try:
                img_gt_recover = np.load(gt_recover_path)
                img_gt_recover = img_gt_recover.squeeze()
            except:
                raise Exception("gt_recover path {} not working".format(gt_recover_path))

            # 加载dgain信息
            dgain_path = self.paths[index]['dgain_path']
            with open(dgain_path, 'r') as file:
                prompt = file.read()
            dgain_info = extract_info_from_dgain_prompt(prompt)
            dgain = dgain_info['dgain']
            gamma = dgain_info['gamma']
        
        # 统一的图像处理流程（不论是LMDB还是disk模式）
        # 对GT图像进行log tone mapping
        img_gt = log_tone_mapping(img_gt, hdr_max=1000)

        # 对LQ图像进行逆tone mapping然后再log tone mapping
        img_lq = inverse_custom_tone_mapping(img_lq, dgain, gamma, max_value=1000)
        img_lq = log_tone_mapping(img_lq, hdr_max=1000)

        # 对GT_RECOVER图像进行log tone mapping（假设它已经是HDR格式）
        img_gt_recover = img_gt_recover / 1000.0
        img_gt_recover = cv2.cvtColor(img_gt_recover, cv2.COLOR_RGB2BGR)

        # 训练时的数据增强
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            # 应用padding
            img_gt, img_lq, img_gt_recover = padding_triple(img_gt, img_lq, img_gt_recover, gt_size)

            # 应用随机裁剪
            img_gt, img_lq, img_gt_recover = paired_random_crop_triple(
                img_gt, img_lq, img_gt_recover, gt_size, scale, gt_path)

            # 应用几何增强
            if self.geometric_augs:
                img_gt, img_lq, img_gt_recover = random_augmentation_triple(img_gt, img_lq, img_gt_recover)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_gt_recover = img2tensor([img_gt, img_lq, img_gt_recover],
                                                    bgr2rgb=True,
                                                    float32=True)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_gt_recover, self.mean, self.std, inplace=True)
        
        # 返回数据：lq和gt_recover作为输入，gt作为目标
        return {
            'lq': img_lq,                    # 网络输入1：LDR图像
            'gt_recover': img_gt_recover,    # 网络输入2：其他网络恢复的HDR图像
            'gt': img_gt,                    # 网络目标：真实HDR图像
            'lq_path': lq_path,
            'gt_path': gt_path,
            'gt_recover_path': gt_recover_path
        }

    def __len__(self):
        return len(self.paths)


"""
配置文件示例：
要使用LDR_HDR_HDR_RECOVER_PairedDataset，在配置文件中需要设置以下参数：

# Disk模式配置示例
datasets:
  train:
    name: TrainDataset
    type: LDR_HDR_HDR_RECOVER_PairedDataset
    dataroot_lq: /path/to/ldr_images          # LDR图像文件夹
    dataroot_gt: /path/to/hdr_ground_truth    # 真实HDR图像文件夹
    dataroot_gt_recover: /path/to/hdr_recover # 其他网络恢复的HDR图像文件夹
    dataroot_dgain: /path/to/dgain_files      # dgain信息文件夹
    filename_tmpl: '{}'
    io_backend:
      type: disk
    gt_size: 512
    geometric_augs: true
    scale: 1
    phase: train

# LMDB模式配置示例
datasets:
  train:
    name: TrainDataset
    type: LDR_HDR_HDR_RECOVER_PairedDataset
    dataroot_lq: /path/to/ldr_images.lmdb     # LDR图像LMDB数据库
    dataroot_gt: /path/to/hdr_gt.lmdb         # 真实HDR图像LMDB数据库
    dataroot_gt_recover: /path/to/hdr_recover.lmdb  # 恢复HDR图像LMDB数据库
    dataroot_dgain: /path/to/dgain_info.lmdb  # dgain信息LMDB数据库
    filename_tmpl: '{}'
    io_backend:
      type: lmdb
      db_paths: [lq_path, gt_path, gt_recover_path, dgain_path]  # 会自动设置
      client_keys: [lq, gt, gt_recover, dgain]                  # 会自动设置
    gt_size: 512
    geometric_augs: true
    scale: 1
    phase: train
    
注意事项：
1. Disk模式：四个数据文件夹中的文件必须按相同的基础文件名对应
2. LMDB模式：四个LMDB数据库中的键必须对应，dgain信息以UTF-8文本格式存储
3. 数据集返回三个图像：
   - lq: LDR输入图像
   - gt_recover: 其他网络恢复的HDR图像  
   - gt: 真实HDR图像（用于计算loss）
4. 网络的输入是lq和gt_recover，输出与gt比较计算loss
5. LMDB模式通常具有更快的I/O性能，特别适合大规模训练
"""




