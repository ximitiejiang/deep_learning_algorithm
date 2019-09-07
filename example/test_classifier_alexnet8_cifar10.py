#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
from model.runner_lib import Runner



def main():
    runner = Runner(cfg_path = '../model/cfg_classifier_alexnet8_mnist.py')
#    runner.train()
    runner.evaluate()
    
    
if __name__ == "__main__":
    main()