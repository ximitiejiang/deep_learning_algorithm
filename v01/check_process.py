#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:11:01 2019

@author: ubuntu
"""
from utils.global_var import gl
gl.set_value("abc", 1000)

from model.one_stage_detector import OneStageDetector
from dataset.voc_dataset import VOCDataset
import sys,os
import numpy as np
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)

    
from utils.tester import TestImg, TestVideo, TestDataset
from utils.support import IO, DRAW

model_class = OneStageDetector
dataset_name = 'voc'
img_path = './data/misc/test14.jpg'
# support jpg/png list...
img_paths = ['./data/misc/test.jpg',
             './data/misc/test11.jpg',
             './data/misc/test12.jpg',
             './data/misc/test13.jpg']

video_path = './data/misc/challenge.mp4'

# ssd300    
config_file = './config/cfg_ssd300_vgg16_voc.py'
weights_path = './weights/myssd/weight_4imgspergpu/epoch_24.pth'
out_file = './weights/myssd/weight_4imgspergpu/results_24.pkl'

# retinanet
#config_file = './config/cfg_retinanet_r50_fpn_voc.py'
#weights_path = './weights/myretinanet/4imgpergpu/epoch_16.pth'
#out_file = './weights/myretinanet/4imgpergpu/results_16.pkl'


#testimg, testcam, testvideo, testdataset = (1,0,0,0)  # choose test mode: 1 means on, 0 means off

#var_list = IO.loadvar('./work_dirs/temp/ssd_test.pkl')       # 加载变量 bbox, score
#DRAW.draw_img_bbox(img_path, var_list[0][2000:2012])         # 绘制图片和bbox
#DRAW.draw_hist(var_list[1][4534])                               # 绘制score分布(8732,21)                
    
test_img = TestImg(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
test_img.run(img_path)

# 实验：基于某张测试图片，查看nms之后的输出(包括score过滤，nms过滤)bbox是否合理：
# 结果：有2个问题，第一不同特征层的bbox有重复检出的问题，第二小物体检测精度不高
#scale_factor = np.array([0.18518518, 0.2777778 , 0.18518518, 0.2777778 ], dtype=np.float32)
scale_factor = np.array([1,1,1,1],dtype=np.float32)
out_list = IO.loadvar('./work_dirs/temp/after_score_thr.pkl')  # bbox, score
TestImg.inside_imshow(img_path, bboxes = out_list[0], scale_factor=scale_factor)



