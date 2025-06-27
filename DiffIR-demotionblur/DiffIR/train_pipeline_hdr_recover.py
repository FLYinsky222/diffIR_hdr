#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDR恢复任务专用训练流程脚本
支持LDR + GT_RECOVER -> HDR的图像恢复任务
使用其他网络的恢复结果作为辅助输入
"""
import datetime
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
import numpy as np
import random


def init_tb_loggers(opt):
    """初始化TensorBoard和Wandb日志记录器"""
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    """创建训练和验证数据加载器"""
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('HDR恢复训练统计信息:'
                        f'\n\t训练图像数量: {len(train_set)}'
                        f'\n\t数据集放大比例: {dataset_enlarge_ratio}'
                        f'\n\t每GPU批次大小: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\t世界大小(GPU数量): {opt["world_size"]}'
                        f'\n\t每轮次所需迭代数: {num_iter_per_epoch}'
                        f'\n\t总轮次: {total_epochs}; 总迭代数: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'{dataset_opt["name"]}中的验证图像/文件夹数量: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'数据集阶段 {phase} 未被识别。')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    """加载恢复状态文件"""
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline_hdr_recover(root_path):
    """HDR恢复任务专用训练流程主函数"""
    # 解析选项，设置分布式设置，设置随机种子
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 如有必要，加载恢复状态
    resume_state = load_resume_state(opt)
    # 为实验和日志记录器创建目录
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # 将yml文件复制到实验根目录
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # 警告：不应在上述代码中使用get_root_logger，包括调用的函数
    # 否则日志记录器将无法正确初始化
    log_file = osp.join(opt['path']['log'], f"train_hdr_recover_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # 初始化wandb和tb日志记录器
    tb_logger = init_tb_loggers(opt)

    # 创建训练和验证数据加载器
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # 创建模型
    model = build_model(opt)
    if resume_state:  # 恢复训练
        model.resume_training(resume_state)  # 处理优化器和调度器
        logger.info(f"从轮次: {resume_state['epoch']}, 迭代: {resume_state['iter']}恢复训练。")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # 创建消息日志记录器（格式化输出）
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # 数据加载器预取器
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'使用 {prefetch_mode} 预取数据加载器')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('请为CUDAPrefetcher设置pin_memory=True。')
    else:
        raise ValueError(f"错误的prefetch_mode {prefetch_mode}。支持的选项有: None, 'cuda', 'cpu'。")

    # HDR恢复任务的训练配置
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    gt_size = opt['datasets']['train'].get('gt_size', 512)
    
    logger.info(f"HDR恢复训练配置:")
    logger.info(f"  - 批次大小: {batch_size}")
    logger.info(f"  - 图像尺寸: {gt_size}x{gt_size}")
    logger.info(f"  - 总迭代数: {total_iters}")
    logger.info(f"  - 网络输入: LDR图像 + 其他网络恢复的HDR图像")
    logger.info(f"  - 网络输出: 优化后的HDR图像")
    logger.info(f"  - 损失计算: 网络输出 vs 真实HDR图像")

    # 开始训练
    logger.info(f'开始HDR恢复训练，从轮次: {start_epoch}, 迭代: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
                
            # 更新学习率
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # HDR恢复任务的数据处理
            # 数据流程: LQ (LDR) + GT_RECOVER (其他网络的HDR恢复) -> 网络输出 -> 与GT (真实HDR) 计算loss
            lq = train_data['lq']                      # LDR输入图像
            gt = train_data['gt']                      # 真实HDR图像（用于计算loss）
            gt_recover = train_data['gt_recover']      # 其他网络恢复的HDR图像（网络输入）
            
            # 可选：获取dgain信息（如果数据集提供）
            dgain_info = train_data.get('dgain', None)
            
            # 准备训练数据：网络需要lq、gt_recover作为输入，gt作为目标
            feed_data = {
                'lq': lq,                    # 网络输入1: LDR图像
                'gt_recover': gt_recover,    # 网络输入2: 其他网络恢复的HDR图像
                'gt': gt                     # 网络目标: 真实HDR图像
            }
            if dgain_info is not None:
                feed_data['dgain'] = dgain_info

            # 执行训练
            model.feed_data(feed_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            
            if current_iter == 1:
                # 在msg_logger中重置开始时间以获得更准确的eta_time
                # 在恢复模式下不工作
                msg_logger.reset_start_time()
                
            # 记录日志
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # 保存模型和训练状态
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('保存模型和训练状态。')
                model.save(epoch, current_iter)

            # 验证
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('多个验证数据集*仅*由SRModel支持。')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # 迭代结束

    # 轮次结束

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'HDR恢复训练结束。消耗时间: {consumed_time}')
    logger.info('保存最新模型。')
    model.save(epoch=-1, current_iter=-1)  # -1表示最新
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline_hdr_recover(root_path) 