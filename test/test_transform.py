#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:09:14 2019

@author: ubuntu
"""
from utils.prepare_training import get_config, get_dataset

import sys, os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)
"""显示区别
常规数据集出来的图片都是hwc,bgr格式
1. plt.imshow(), 支持hwc, rgb
2. cv2.imshow(), 支持hwc, bgr
"""
cfg_path = '/home/ubuntu/suliang_git/deep_learning_algorithm/demo/retinface_widerface/cfg_detector_retinaface_widerface.py'
cfg = get_config(cfg_path)
trainset = get_dataset(cfg.trainset, cfg.transform)
tmp1 = trainset[91]
img = tmp1['img']
label = tmp1['gt_labels']
bbox = tmp1['gt_bboxes']
ldmk = tmp1['gt_landmarks']
from utils.transform import transform_inv
class_names = trainset.CLASSES
label = label - 1 # 恢复从0为起点，从而跟CLASS匹配
transform_inv(img, bbox, label, ldmk, mean=cfg.transform.img_params.mean, 
              std=cfg.transform.img_params.std, class_names=class_names, show=True)
