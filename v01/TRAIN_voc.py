#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:42:09 2019

@author: ubuntu
"""
import sys,os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)        
import logging
from torch.utils.data import DataLoader
import torch.distributed as dist
from collections import OrderedDict
import torch
from functools import partial

from utils.runner.runner import Runner
from dataset.sampler import GroupSampler  # 用于dataloader采样定义
from model.one_stage_detector import OneStageDetector
from model.parallel.data_parallel import NNDataParallel
from model.parallel.collate import collate
from utils.config import Config
from dataset.voc_dataset import VOCDataset
from dataset.utils import get_dataset
    
def get_dist_info():
    if torch.__version__ < '1.0':     
        initialized = dist._initialized    # pytorch 0.4.1
    else:
        initialized = dist.is_initialized()  # pytorch 1.0
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_root_logger(log_level=logging.INFO):
    # 先创建logger
    logger = logging.getLogger()
    if not logger.hasHandlers():
        # 进行logging基本设置format/level
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
        
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    return logger

def batch_processor(model, data, train_mode):
    """创建一个基础batch process，用来搭配runner模块进行整个计算框架的组成
    1. 计算损失
    2. 解析损失并组合输出
    Args:
        model(Module)
        data()
    Returns:
        
    """
    losses = model(**data)
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))
    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    
    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()
        
    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs  
  
def train(cfg_path, dataset_class):
    """借用mmcv的Runner框架进行训练，包括里边的hooks作为lr更新，loss计算的工具
    1. dataset的数据集输出打包了img/gt_bbox/label/，采用DataContainer封装
    2. Dataloader的default_collate用定制collate替换，从而支持dataset的多类型数据
    3. DataParallel外壳用定制MMDataparallel替换，从而支持DataContainer
    """
    # 初始化2个默认选项
    distributed = False
    parallel = True      # 必须设置成dataparallel模式，否则data container无法拆包(因为data container的拆包是在data parallel的scatter函数中进行的)
    
    # get cfg
    cfg = Config.fromfile(cfg_path)
    
    # set backends
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # get logger
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('DataParallel training: {}'.format(parallel))
    
    # %% build model & detector
    model = OneStageDetector(cfg)
#    model = OneStageDetector(cfg)
    if not parallel:
        model = model.cuda()
    else:
        model = NNDataParallel(model, device_ids = range(cfg.gpus)).cuda()
    
    # prepare data & dataloader
    # Runner要求dataloader放在list里: 使workflow里每个flow对应一个dataloader
    dataset = get_dataset(cfg.data.train, dataset_class)
    batch_size = cfg.gpus * cfg.data.imgs_per_gpu
    num_workers = cfg.gpus * cfg.data.workers_per_gpu
    dataloader = [DataLoader(dataset, 
                            batch_size=batch_size, 
                            sampler = GroupSampler(dataset, cfg.data.imgs_per_gpu),
                            num_workers=num_workers,
                            collate_fn=partial(collate, samples_per_gpu=cfg.data.imgs_per_gpu),
                            pin_memory=False)] 
    
    # %% 用runner进行训练
    # define runner and running type(1.resume, 2.load, 3.train/test)
    runner = Runner(model, 
                    batch_processor, 
                    cfg.optimizer, 
                    cfg.work_dir, 
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config,
                                   cfg.optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config)
    if cfg.resume_from:  # 恢复训练: './work_dirs/ssd300_voc/latest.pth'
        runner.resume(cfg.resume_from, map_location = lambda storage, loc: storage)
    elif cfg.load_from:  # 加载参数进行测试
        runner.load_checkpoint(cfg.load_from)
    # 开始训练: 采用workflow来区分train还是test
    runner.run(dataloader, cfg.workflow, cfg.total_epochs)
    
    
if __name__ == '__main__':
    # ssd300
#    cfg_path = 'config/cfg_ssd300_vgg16_voc.py'
    
    # ssd512
#    cfg_path = 'config/cfg_ssd512_vgg16_voc.py' 
    
    # ssd512 + mlfpn
#    cfg_path = 'config/cfg_m2det512_vgg16_mlfpn_voc.py'
    
    # retinanet 
    cfg_path = 'config/cfg_retinanet_r50_fpn_voc.py'
    
    train(cfg_path, VOCDataset)
    