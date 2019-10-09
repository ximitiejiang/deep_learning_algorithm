#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:29:33 2019

@author: ubuntu
"""
import cv2
from model.runner_lib import Runner
from utils.evaluation import eval_dataset_det, DetPredictor
from utils.tools import parse_log
from utils.dataset_classes import get_classes
from utils.visualization import vis_all_opencv, vis_cam

def train_ssd_widerface(cfg_path, resume_from):

    runner = Runner(cfg_path = cfg_path, resume_from=resume_from)
    runner.train()    
    
    
    
if __name__ == "__main__":
    
    task = 'eval'
    cfg_path = './cfg_detector_ssdvgg16_widerface.py'
    
    if task == 'train':  # 模型训练
        train_ssd_widerface(cfg_path,
                            resume_from = '/home/ubuntu/mytrain/ssd_vgg_widerface/epoch_9.pth')
    
    if task == 'eval':  # 数据集评估
        parse_log('/home/ubuntu/mytrain/ssd_vgg_widerface/20191008_211622.log')
        
#        eval_dataset_det(cfg_path=cfg_path,
#                         load_from = '/home/ubuntu/mytrain/ssd_vgg_voc/epoch_11.pth',
#                         load_device='cuda')
    
    if task == 'load':  # 已有数据集评估文件，重新载入进行评估
        eval_dataset_det(cfg_path=cfg_path,
                         load_from = '/home/ubuntu/mytrain/ssd_vgg_widerface/epoch_9.pth',
                         load_device='cuda',
                         result_file='/home/ubuntu/mytrain/ssd_vgg_voc/20190928_084133_eval_result.pkl')
    
    if task == 'test':  # 测试单张图或多张图的结果
        img = cv2.imread('../test/nba.jpg')
        predictor = DetPredictor(cfg_path,                         
                              load_from = '/home/ubuntu/mytrain/ssd_vgg_widerface/epoch_9.pth',
                              load_device='cuda')
        for results in predictor([img]):
            vis_all_opencv(*results, class_names=get_classes('widerface'), score_thr=0.2)
    
    # 针对widerface的训练结果模型，对小脸比较有效，但对大脸效果很差，因为数据集里边大部分都是小脸，大脸数据太少。
    # 所以靠近摄像头反而检不出来，远离摄像头就能检出来。
    if task == 'video': 
        src = 0  # src可以等于int(cam_id), str(video path), list(img_list)
        predictor = DetPredictor(cfg_path,                         
                              load_from = '/home/ubuntu/mytrain/ssd_vgg_widerface/epoch_9.pth',
                              load_device='cuda')
        vis_cam(src, predictor, class_names=get_classes('widerface'), score_thr=0.2)
            