#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:54:36 2019

@author: ubuntu
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(launcher='pytorch', backend='nccl', **kwargs):
    # 基于pytorch的底层驱动来初始化分布式
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend, **kwargs)
        
def dist_train(model, dataset, cfg):
    pass