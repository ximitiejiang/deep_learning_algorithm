#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""

from model.runner_lib import Runner
from utils.evaluation import eval_dataset_cls
from utils.tools import parse_args

def train_resnet(cfg_path):
    runner = Runner(cfg_path = cfg_path, resume_from=None)
    runner.train()
    
    
if __name__ == "__main__":
    task = 'train'
    cfg_path = './cfg_classifier_resnet18_cifar10.py'
    
    args = parse_args()
    if args.task is None:
        args.task = task
    if args.config is None:
        args.config = cfg_path
        
    if args.task == 'train':
        train_resnet(cfg_path = args.config)