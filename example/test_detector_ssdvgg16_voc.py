#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
from model.runner_lib import Runner
from utils.evaluation import eval_dataset_det
#import sys, os
#sys.path.insert(0, os.path.abspath('..'))

def train_ssd(cfg_path):
    
    runner = Runner(cfg_path = cfg_path)
    runner.train()    
    
    
    
if __name__ == "__main__":
    
    training = False
    
    cfg_path = './cfg_detector_ssdvgg16_voc.py'
    if training:
        train_ssd(cfg_path)
    else:
        eval_dataset_det(cfg_path=cfg_path,
                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_9.pth',
                         load_device='cuda')
    
    