#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
import torch
from model.runner_lib import Runner
from utils.evaluation import eval_dataset_cls


def test_alexnet8():
    runner = Runner(cfg_path = './cfg_classifier_alexnet8_cifar10.py')
    runner.train()
    
    # 另一种方式测试: 脱离runner, 但需要cfg中定义load_from
#    eval_dataset_cls(cfg_path = './cfg_classifier_alexnet8_cifar10.py')
    
    
if __name__ == "__main__":
    test_alexnet8()