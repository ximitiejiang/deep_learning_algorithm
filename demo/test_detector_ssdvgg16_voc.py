#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
import cv2
from model.runner_lib import Runner
from utils.evaluation import eval_dataset_det, Predictor
from utils.tools import parse_log, timer
from utils.dataset_classes import get_classes
from utils.visualization import vis_all_opencv, vis_all_pyplot, vis_cam

def train_ssd(cfg_path):
    
    runner = Runner(cfg_path = cfg_path)
    runner.train()    
    
    
    
if __name__ == "__main__":
    
    task = 'test'
    cfg_path = './cfg_detector_ssdvgg16_voc.py'
    
    if task == 'train':  # 模型训练
        train_ssd(cfg_path)
    
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
    
    if task == 'test':  # 测试单张图或多张图的结果
        img = cv2.imread('../test/4.jpg')
        predictor = Predictor(cfg_path,                         
                              load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                              load_device='cpu')
        with timer('predict one img'):
            for results in predictor([img]):
                vis_all_pyplot(*results, class_names=get_classes('voc'), score_thr=0.2)
    
    if task == 'video': # 测试视频预测结果：注意方法稍有不同，vis_cam需要传入一个predictor
        src = 0  # src可以等于int(cam_id), str(video path), list(img_list)
        predictor = Predictor(cfg_path,                         
                              load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
                              load_device='cpu')
        vis_cam(src, predictor, class_names=get_classes('voc'), score_thr=0.2)
            