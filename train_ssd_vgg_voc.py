#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 12:18:46 2019

@author: ubuntu
"""
from collections import OrderedDict

import torch
#import argparse
from runner.runner import Runner
from models.data_parallel import MMDataParallel
import logging
#from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
#                        CocoDistEvalRecallHook, CocoDistEvalmAPHook)
from datasets.build_loader import build_dataloader
#from mmdet.models import RPN
from models.builder import build_detector
from datasets.voc_dataset import get_datasets
from config.config import Config

"""train代码改自mmdet.apis/train.py 和tools.train.py"""

def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
#    rank, _ = get_dist_info()
#    if rank != 0:
#        logger.setLevel('ERROR')
    return logger


def parse_losses(losses):
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

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
#    if distributed:
#        _dist_train(model, dataset, cfg, validate=validate)
#    else:
#        _non_dist_train(model, dataset, cfg, validate=validate)
    
# 合并了non_distribution 和 train_detector()    
    data_loaders = [
    build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        cfg.gpus,
        dist=False)
    ]
      
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


#def _dist_train(model, dataset, cfg, validate=False):
#    # prepare data loaders
#    data_loaders = [
#        build_dataloader(
#            dataset,
#            cfg.data.imgs_per_gpu,
#            cfg.data.workers_per_gpu,
#            dist=True)
#    ]
#    # put model on gpus
#    model = MMDistributedDataParallel(model.cuda())
#    # build runner
#    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
#                    cfg.log_level)
#    # register hooks
#    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
#    runner.register_training_hooks(cfg.lr_config, optimizer_config,
#                                   cfg.checkpoint_config, cfg.log_config)
#    runner.register_hook(DistSamplerSeedHook())
#    # register eval hooks
#    if validate:
#        if isinstance(model.module, RPN):
#            # TODO: implement recall hooks for other datasets
#            runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
#        else:
#            if cfg.data.val.type == 'CocoDataset':
#                runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))
#            else:
#                runner.register_hook(DistEvalmAPHook(cfg.data.val))
#
#    if cfg.resume_from:
#        runner.resume(cfg.resume_from)
#    elif cfg.load_from:
#        runner.load_checkpoint(cfg.load_from)
#    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


#def _non_dist_train(model, dataset, cfg, validate=False):
#    # prepare data loaders
#    data_loaders = [
#        build_dataloader(
#            dataset,
#            cfg.data.imgs_per_gpu,
#            cfg.data.workers_per_gpu,
#            cfg.gpus,
#            dist=False)
#    ]
#      
#    # put model on gpus
#    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
#    # build runner
#    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
#                    cfg.log_level)
#    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
#                                   cfg.checkpoint_config, cfg.log_config)
#
#    if cfg.resume_from:
#        runner.resume(cfg.resume_from)
#    elif cfg.load_from:
#        runner.load_checkpoint(cfg.load_from)
#    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

#def parse_args():
#    parser = argparse.ArgumentParser(description='Train a detector')
#    parser.add_argument('config', help='train config file path')
#    parser.add_argument('--work_dir', help='the dir to save logs and models')
#    parser.add_argument(
#        '--resume_from', help='the checkpoint file to resume from')
#    parser.add_argument(
#        '--validate',
#        action='store_true',
#        help='whether to evaluate the checkpoint during training')
#    parser.add_argument(
#        '--gpus',
#        type=int,
#        default=1,
#        help='number of gpus to use '
#        '(only applicable to non-distributed training)')
#    parser.add_argument('--seed', type=int, default=None, help='random seed')
#    parser.add_argument(
#        '--launcher',
#        choices=['none', 'pytorch', 'slurm', 'mpi'],
#        default='none',
#        help='job launcher')
#    parser.add_argument('--local_rank', type=int, default=0)
#    args = parser.parse_args()
#
#    return args

def main():
#    args = parse_args()
#    cfg = Config.fromfile(args.config)
    cfg = Config.fromfile('config/cfg_ssd300_voc.py')  # 修改成自定义基于voc的cfg
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
#    # update configs according to CLI args
#    if args.work_dir is not None:
#        cfg.work_dir = args.work_dir
#    if args.resume_from is not None:
#        cfg.resume_from = args.resume_from
#    cfg.gpus = args.gpus
    cfg.gpus = 1    # 增加gpus的参数，而不是从args获得，
                    # 其中设置gups=1用来调试，否则进入dataparallel模型的thread模块，无法进入具体module
    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(config=cfg.text)

#    # init distributed env first, since logger depends on the dist info.
#    if args.launcher == 'none':
#        distributed = False
#    else:
#        distributed = True
#        init_dist(args.launcher, **cfg.dist_params)
    distributed = False  # 增加分布式训练定义为False，采用并行计算训练

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
#    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
#    if args.seed is not None:
#        logger.info('Set random seed to {}'.format(args.seed))
#        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_dataset = get_datasets(cfg.data.train)
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=False,
        logger=logger)
   
if __name__ == '__main__':
    main()
