#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""

from model.runner_lib import Runner
from utils.evaluation import eval_dataset_cls


def train_resnet():
    runner = Runner(cfg_path = './cfg_classifier_resnet18_cifar10.py')
    runner.train()
    
    
if __name__ == "__main__":
    task = 'train'
    
    if task == 'train':
        train_resnet()