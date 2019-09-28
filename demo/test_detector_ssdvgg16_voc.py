#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
import cv2
from model.runner_lib import Runner
from utils.evaluation import eval_dataset_det, predict_one_img_det
from utils.tools import parse_log
from utils.dataset_classes import get_classes

def train_ssd(cfg_path):
    
    runner = Runner(cfg_path = cfg_path)
    runner.train()    
    
    
    
if __name__ == "__main__":
    
    task = 'test'
    cfg_path = './cfg_detector_ssdvgg16_voc.py'
    
    if task == 'train':
        train_ssd(cfg_path)
    
    
    if task == 'eval':
        parse_log('/home/ubuntu/mytrain/ssd_vgg_voc/20190926_181047.log')
        
        eval_dataset_det(cfg_path=cfg_path,
                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                         load_device='cuda')
    
    if task == 'load':
        
        eval_dataset_det(cfg_path=cfg_path,
                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                         load_device='cuda',
                         result_file='/home/ubuntu/mytrain/ssd_vgg_voc/20190928_084133_eval_result.pkl')
    
    if task == 'test':
        img = cv2.imread('../test/1.jpg')
        predict_one_img_det(img, cfg_path,                         
                            load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                            load_device='cuda',
                            class_names = get_classes('voc'))