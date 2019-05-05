#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:26:28 2019

@author: ubuntu
"""

from model.one_stage_detector import OneStageDetector
from dataset.voc_dataset import VOCDataset
import sys,os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)

    
from utils.tester import TestImg, TestVideo, TestDataset

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

testimg, testcam, testvideo, testdataset = (0,0,1,0)  # choose test mode: 1 means on, 0 means off

if testimg:        
    test_img = TestImg(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
    test_img.run(img_paths)

elif testcam:
    test_video = TestVideo(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
    test_video.run(source=0)

elif testvideo:
    test_video = TestVideo(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
    test_video.run(source=video_path)
    
elif testdataset:
    test_dataset = TestDataset(config_file, model_class, weights_path, dataset_name='voc', device = 'cuda:0', out_file=out_file)
    test_dataset.run(VOCDataset)
    