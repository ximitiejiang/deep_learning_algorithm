#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:26:28 2019

@author: ubuntu
"""

from model.one_stage_detector import OneStageDetector
import sys,os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)

    
from utils.test_class import TestImg, TestVideo

model_class = OneStageDetector
dataset_name = 'voc'
img_path = './data/misc/test.jpg'
video_path = './data/misc/challenge.mp4'    
config_file = './config/cfg_ssd300_vgg16_voc.py'
weights_path = './weights/myssd/weight_4imgspergpu/epoch_24.pth'
        
#test_img = TestImg(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
#test_img.run(img_path)

test_video = TestVideo(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
test_video.run(source=0)

#test_video = TestVideo(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
#test_video.run(source=video_path)
    