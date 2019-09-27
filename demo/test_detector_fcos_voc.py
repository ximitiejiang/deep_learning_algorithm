#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
from model.runner_lib import Runner
from utils.evaluation import eval_dataset_det
from utils.tools import parse_log

def train_fcos(cfg_path):
    
    runner = Runner(cfg_path = cfg_path)
    runner.train()    
    
    
    
if __name__ == "__main__":
    
    task = 'eval'
    
    cfg_path = './cfg_detector_fcos_resnet50_voc.py'
    
    if task == 'train':
        train_fcos(cfg_path)
        
        
    
    if task == 'eval':
        parse_log('/home/ubuntu/mytrain/ssd_vgg_voc/20190926_181047.log')
        
        eval_dataset_det(cfg_path=cfg_path,
                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                         load_device='cuda')
    
    if task == 'load':
        
        
        eval_dataset_det(cfg_path=cfg_path,
                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_9.pth',
                         load_device='cuda',
                         result_file='/home/ubuntu/mytrain/ssd_vgg_voc/eval_result.pkl')
    