#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
import cv2
from model.runner_lib import Runner
from utils.prepare_training import get_config, get_dataset
from utils.evaluation import SegPredictor
from utils.tools import parse_log
from utils.dataset_classes import get_classes
from utils.visualization import vis_cam

def train_fcn(cfg_path):
    
    runner = Runner(cfg_path = cfg_path)
    runner.train()    
    
    
    
if __name__ == "__main__":
    
    task = 'test'
    cfg_path = './cfg_segmentator_fcn_vgg16_voc.py'
    
    if task == 'pre':
        cfg = get_config(cfg_path)
        trainset = get_dataset(cfg.trainset, cfg.transform)
        data = trainset[1]

    
    if task == 'train':  # 模型训练
        train_fcn(cfg_path)
    
#    if task == 'eval':  # 数据集评估
#        parse_log('/home/ubuntu/mytrain/ssd_vgg_voc/20190926_181047.log')
#        
#        eval_dataset_det(cfg_path=cfg_path,
#                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
#                         load_device='cuda')
#    
#    if task == 'load':  # 已有数据集评估文件，重新载入进行评估
#        eval_dataset_det(cfg_path=cfg_path,
#                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
#                         load_device='cuda',
#                         result_file='/home/ubuntu/mytrain/ssd_vgg_voc/20190928_084133_eval_result.pkl')
    
    if task == 'test':  # 测试单张图或多张图的结果： cpu上0.649 sec， gpu上0.388 sec
        img = cv2.imread('../test/009779.jpg')
        predictor = SegPredictor(cfg_path,                         
                                 load_from = '/home/ubuntu/mytrain/fcn_vgg_voc/epoch_9.pth',
                                 load_device='cpu')
        for result in predictor([img]):
            cv2.imshow('seg', result)
    
    if task == 'video': # 测试视频预测结果：注意方法稍有不同，vis_cam需要传入一个predictor
        src = 0  # src可以等于int(cam_id), str(video path), list(img_list)
        predictor = SegPredictor(cfg_path,                         
                                 load_from = '/home/ubuntu/mytrain/fcn_vgg_voc/epoch_9.pth',
                                 load_device='cuda')
        vis_cam(src, predictor, class_names=get_classes('voc'), score_thr=0.2)
            