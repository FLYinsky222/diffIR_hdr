import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from DiffIR.models import lr_scheduler as lr_scheduler

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_, gt_recover):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
        gt_recover = lam * gt_recover + (1-lam) * gt_recover[r_index, :]
    
        return target, input_, gt_recover

    def __call__(self, target, input_, gt_recover):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_, gt_recover = self.augments[augment](target, input_, gt_recover)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_, gt_recover = self.augments[augment](target, input_, gt_recover)
        return target, input_, gt_recover

#@MODEL_REGISTRY.register() 是一种典型的 装饰器注册机制，常见于 DiffIR、BasicSR 等框架中，用于 自动注册模型类到某个模型字典中，便于通过字符串配置动态创建模型。
@MODEL_REGISTRY.register()
class DiffIRS1HDRModel(SRModel):
    """
    DiffIR S1 model for HDR recovery with gt_recover input.
    
    The network takes LQ (LDR image) and GT_RECOVER (HDR recovered by other networks) as inputs,
    and outputs HDR image that is compared with GT for loss calculation.
    
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(DiffIRS1HDRModel, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
    
    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingWarmupRestarts(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLRWithRestart(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def feed_data(self, data):
        """Feed data to the model.
        
        Args:
            data (dict): Input data containing 'lq', 'gt_recover', and 'gt'.
                - lq: LDR input image
                - gt_recover: HDR image recovered by other networks
                - gt: Ground truth HDR image
        """
        self.lq = data['lq'].to(self.device)
        
        # 处理gt_recover数据（新增）
        if 'gt_recover' in data:
            self.gt_recover = data['gt_recover'].to(self.device)
        else:
            # 向后兼容：如果没有gt_recover，使用lq代替（虽然不推荐）
            print("Warning: gt_recover not found in data, using lq as fallback")
            self.gt_recover = self.lq
            
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 数据增强：现在需要同时处理三个数据
        if self.is_train and self.mixing_flag:
            self.gt, self.lq, self.gt_recover = self.mixing_augmentation(self.gt, self.lq, self.gt_recover)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(DiffIRS1HDRModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def test(self):
        """Testing function.
        
        The network takes lq and gt_recover as inputs.
        """
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # 网络输入：lq (LDR图像) 和 gt_recover (其他网络恢复的HDR)
                self.output = self.net_g_ema(self.lq, self.gt_recover)
        else:
            self.net_g.eval()
            with torch.no_grad():
                # 网络输入：lq (LDR图像) 和 gt_recover (其他网络恢复的HDR)
                self.output = self.net_g(self.lq, self.gt_recover)
            self.net_g.train()

    def optimize_parameters(self, current_iter):
        """Optimization step.
        
        The network takes lq and gt_recover as inputs, outputs HDR image,
        which is compared with gt for loss calculation.
        """
        self.optimizer_g.zero_grad()
        
        # 网络前向传播：输入lq和gt_recover，输出HDR图像
        self.output, _ = self.net_g(self.lq, self.gt_recover)

        l_total = 0
        loss_dict = OrderedDict()
        
        # pixel loss: 网络输出与真实HDR (gt) 计算loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
        # perceptual loss: 网络输出与真实HDR (gt) 计算loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
