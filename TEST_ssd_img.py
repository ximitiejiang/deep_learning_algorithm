#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:26:28 2019

@author: ubuntu
"""

import torch
import cv2
import numpy as np

from model.checkpoint import load_checkpoint
from utils.config import Config
from dataset.transforms import ImageTransform
from dataset.class_names import get_classes
from dataset.utils import vis_bbox
from model.one_stage_detector import OneStageDetector


def test_img(img_path, config_file, weights_path, class_name='voc', device = 'cuda:0'):
    """测试单张图片：相当于恢复模型和参数后进行单次前向计算得到结果
    注意由于没有dataloader，所以送入model的数据需要手动合成img_meta
    1. 模型输入data的结构：需要手动配出来
    2. 模型输出result的结构：
    Args:
        img(array): (h,w,c)-bgr
        config_file(str): config文件路径
        device(str): 'cpu'/'cuda:0'/'cuda:1'
        class_name(str): 'voc' or 'coco'
    """
    # 1. 配置文件
    cfg = Config.fromfile(config_file)
    cfg.model.pretrained = None     # eval模式不再加载backbone的预训练参数，因为都在后边的checkpoints里边包含了。通过model load checkpoint统一来操作。
    # 2. 模型
    model = OneStageDetector(cfg)
    _ = load_checkpoint(model, weights_path)
    model = model.to(device)
    model.eval()             
    
    # 3. 图形/数据变换
    img = cv2.imread(img_path)
    img_transform = ImageTransform(**cfg.img_norm_cfg)  # 测试阶段只采用ImageTransform这个基础变换
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img, 
        scale= cfg.data.test.img_scale, 
        keep_ratio=False)  
        # ssd要求输入必须300*300，所以keep_ratio必须False，否则可能导致图片变小输出最后一层size计算为负
    img = torch.tensor(img).to(device).unsqueeze(0) # 应该从(3,800,1216) to tensor(1,3,800,1216)
    
    # 4. 数据包准备
    img_meta = [dict(ori_shape=ori_shape,
                     img_shape=img_shape,
                     pad_shape=pad_shape,
                     scale_factor = scale_factor,
                     flip=False)]

    data = dict(img=[img], img_meta=[img_meta])
    
    # 5. 结果计算: result    
    with torch.no_grad():
        result = model(**data, return_loss=False, rescale=True)
    # 6. 结果显示
    class_names = get_classes(class_name)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) 
                for i, bbox in enumerate(result)]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    scores = bboxes[:,-1]
    img = cv2.imread(img_path, 1)
    vis_bbox(img.copy(), bboxes, label=labels, score=scores, score_thr=0.4, 
             label_names=class_names,
             instance_colors=None, alpha=1., linewidth=1.5, ax=None)
    

if __name__ == "__main__":     
    
    test_this_img = True
    
    if test_this_img:
        img_path = './data/misc/test14.jpg'    
        config_file = './config/cfg_ssd300_vgg16_voc.py'
        weights_path = './weights/myssd/epoch_24.pth'
        class_name = 'voc'
        test_img(img_path, config_file, weights_path, class_name=class_name)
