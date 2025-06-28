from torch.utils import data as data
from torchvision.transforms.functional import normalize

from DiffIR.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    triple_paths_from_folder,
                                    triple_paths_from_lmdb)
from DiffIR.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from DiffIR.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from basicsr.utils.registry import DATASET_REGISTRY
import random
import numpy as np
import torch
import pickle
import cv2
from DiffIR.data.common_function import extract_info_from_dgain_prompt, log_tone_mapping, inverse_log_tone_mapping, load_ldr_file, load_hdr, custom_tone_mapping, inverse_custom_tone_mapping

@DATASET_REGISTRY.register()
class LDR_HDR_PairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy,ldr_image, etc) and
    GT image pairs.(hdr_image)

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
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
        super(LDR_HDR_PairedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder, self.dgain_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_dgain']  
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            # LMDB模式：支持三个数据源的LMDB数据库
            self.io_backend_opt['db_paths'] = [
                self.lq_folder, 
                self.gt_folder, 
                self.dgain_folder
            ]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'dgain']
            
            # 使用triple_paths_from_lmdb获取三个数据源的路径信息
            self.paths = triple_paths_from_lmdb(
                [self.lq_folder, self.gt_folder, self.dgain_folder], 
                ['lq', 'gt', 'dgain'])
            
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
            # folder模式：使用triple_paths_from_folder处理三个数据源
            self.paths = triple_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.dgain_folder], ['lq', 'gt', 'dgain'],
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
            dgain_path = self.paths[index]['dgain_path']
            
            # 从LMDB中读取数据
            # LQ图像：使用标准图像编码存储，用imfrombytes读取
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            
            # HDR图像：使用pickle序列化存储，用pickle.loads读取
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = pickle.loads(img_bytes)
            
            # 读取dgain信息（以文本形式存储在LMDB中）
            dgain_bytes = self.file_client.get(dgain_path, 'dgain')
            dgain_info = pickle.loads(dgain_bytes)

            dgain = dgain_info['dgain']
            gamma = dgain_info['gamma']
            
        else:
            # Disk模式的数据加载
            # Load gt and lq images. Dimension order: HWC; channel order: BGR;
            # image range: [0, 1], float32.
            gt_path = self.paths[index]['gt_path']

            try:
                img_gt = load_hdr(gt_path)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            lq_path = self.paths[index]['lq_path']

            try:
                img_lq = load_ldr_file(lq_path)
            except:
                raise Exception("lq path {} not working".format(lq_path))

            dgain_path = self.paths[index]['dgain_path']
            with open(dgain_path, 'r') as file:
                prompt = file.read()
            dgain_info = extract_info_from_dgain_prompt(prompt)
            dgain = dgain_info['dgain']
            gamma = dgain_info['gamma']
        
        # 统一的图像处理流程（不论是LMDB还是disk模式）
        img_gt = log_tone_mapping(img_gt, hdr_max=1000)

        img_lq = inverse_custom_tone_mapping(img_lq, dgain, gamma, max_value=1000)

        img_lq = log_tone_mapping(img_lq, hdr_max=1000)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)




