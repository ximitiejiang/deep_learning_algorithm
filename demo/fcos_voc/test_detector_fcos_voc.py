#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
from model.runner_lib import Runner
from utils.evaluation import eval_dataset_det
from utils.tools import parse_log
import cv2
from utils.prepare_training import get_config
from utils.evaluation import DetPredictor
from utils.visualization import vis_all_pyplot, vis_cam
from utils.dataset_classes import get_classes

def train_fcos(cfg, resume_from=None):
    
    runner = Runner(cfg, resume_from=None)
    runner.train()    
    
    
    
if __name__ == "__main__":
    
    task = 'train'
    cfg_path = './cfg_detector_fcos_resnet50_voc.py'
    cfg = get_config(cfg_path)
    
    if task == 'train':  # 模型训练
        train_fcos(cfg=cfg,
                   resume_from = None)
    
    if task == 'eval':  # 数据集评估
        parse_log('/home/ubuntu/mytrain/ssd_vgg_voc/20190926_181047.log')
        
        eval_dataset_det(cfg_path=cfg_path,
                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                         load_device='cuda')
    
    if task == 'load':  # 已有数据集评估文件，重新载入进行评估
        eval_dataset_det(cfg_path=cfg_path,
                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                         load_device='cuda',
                         result_file='/home/ubuntu/mytrain/ssd_vgg_voc/20190928_084133_eval_result.pkl')
    
    if task == 'test':  # 测试单张图或多张图的结果： cpu上0.649 sec， gpu上0.388 sec
        img = cv2.imread('../test/4.jpg')
        predictor = DetPredictor(cfg_path,                         
                                 load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                                 load_device='cuda')
        for results in predictor([img]):
            vis_all_pyplot(*results, class_names=get_classes('voc'), score_thr=0.2)
    
    if task == 'video': # 测试视频预测结果：注意方法稍有不同，vis_cam需要传入一个predictor
        src = 0  # src可以等于int(cam_id), str(video path), list(img_list)
        predictor = DetPredictor(cfg_path,                         
                                 load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                                 load_device='cpu')
        vis_cam(src, predictor, class_names=get_classes('voc'), score_thr=0.2)    