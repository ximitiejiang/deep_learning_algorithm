#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
import os, sys
print('current work path: ', os.getcwd())
print('sys path: ', sys.path)

from model.runner_lib import Runner
from utils.evaluation import eval_dataset_cls
from utils.prepare_training import get_config
from utils.tools import parse_args
import torch.distributed as dist

"""
注意：命令行运行前需要确保把项目根路径加入sys.path(最好通过bashrc添加)
python -m torch.distributed.launch --nproc_per_node 2 --node_rank 0 ./test_classifier_resnet18_cifar10.py --task train
- 当nproc_per_node=2时，会产生2个进程，而如果cfg只指定了1个GPU工作，此时会让该GPU运行两次相同的进程，整个运行时间也double了。
  所以务必要让nproc_per_node的数量跟cfg中gpus保持数量一致。

- 通过提取os.environ['RANK']成功，可以判断分布式launch是成功的，确实可以启动2个进程，但这两个进程都在同一个GPU上运行，却没有分到2个不同GPU上面去。

refer to:
PYTHON -m torch.distributed.launch --nproc_per_node=2 train_cifar10.py $1 --launcher pytorch


setting: 16pic/gpu, 2epochs, resnet18, cifar10,
        1GPU        2GPU(dist)
our     208s        ?
        loss=2.1
        acc=40

mmcv    240s
        loss=1.3
        acc=53.7

"""

def merge_to(args, cfg):
    """把args合并到cfg, 可采用vars()或者__dict_-把namespace转换成dict
    其中cfg为dict, args为namespace
    """
    args = vars(args)
    for key, value in args.items():
        if value is not None:
            cfg[key] = value
    return cfg
    
    

def train_model(args):

    runner = Runner(cfg, resume_from=None)
    runner.train(vis=True)
    
    
if __name__ == "__main__":
    task = 'eval'
    cfg_path = './cfg_classifier_resnet18_cifar10.py'
    
    args = parse_args()    # 传入task, config, launcher, local_rank   
    cfg = get_config(cfg_path)
    cfg = merge_to(args, cfg)
    if os.environ.get('RANK', None) is not None:
        print('dist launched. RANK = %s'%(os.environ['RANK']))
        task = 'train'
    
    if task == 'train':
        train_model(args)
        
    if task == 'eval':
        print('this is eval')