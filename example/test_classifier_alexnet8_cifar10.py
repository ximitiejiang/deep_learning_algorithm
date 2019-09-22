#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""

from model.runner_lib import Runner



def test_alexnet8_cifar10():
    runner = Runner(cfg_path = './cfg_classifier_alexnet8_cifar10.py')
    runner.train()
    runner.val()
    
    
    
if __name__ == "__main__":
    test_alexnet8_cifar10()