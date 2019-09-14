#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""

from model.runner_lib import Runner


    
    
def test_alexnet_antsbees():
    """实验目的：对比alexnet在pretrained weight条件下，以及在从头训练条件下，
    对简单样本(antsbees dataset 200张训练图)的分类效果.
    
    实验结果：
    """
    runner = Runner(cfg_path = '../model/cfg_classifier_alexnet_antsbees.py')
    runner.train()
    runner.evaluate()
    
    
if __name__ == "__main__":
    test_alexnet_antsbees()