#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:26:28 2019

@author: ubuntu
"""

import torch
import cv2
import numpy as np
import time

from model.checkpoint import load_checkpoint
from utils.config import Config
from dataset.transforms import ImageTransform
from dataset.class_names import get_classes
from dataset.utils import vis_bbox
from model.one_stage_detector import OneStageDetector
import sys,os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)

class Tester(object):
    """测试基类，用于进行单图/多图/摄像头测试"""        
    def __init__(self, config_file, model_class, weights_path, 
                 dataset_name='voc', device = 'cuda:0'):
        self.config_file = config_file
        self.model_class = model_class
        self.weights_path = weights_path
        self.class_names = get_classes(dataset_name)
        self.device = device
        
    def init_cfg_model(self):
        """准备cfg,model
        """
        # 1. 配置文件
        cfg = Config.fromfile(self.config_file)
        cfg.model.pretrained = None     # eval模式不再加载backbone的预训练参数，因为都在后边的checkpoints里边包含了。通过model load checkpoint统一来操作。
        # 2. 模型
        model = self.model_class(cfg)
        _ = load_checkpoint(model, self.weights_path)
        model = model.to(self.device)
        model.eval()             
        
        return cfg, model
    
    def preprocess_img(self, cfg, img, transformer=None):
        ori_shape = img.shape
        if transformer is None:
            transformer = ImageTransform(**cfg.img_norm_cfg)
            
        img, img_shape, pad_shape, scale_factor = transformer(
            img, 
            scale= cfg.data.test.img_scale, 
            keep_ratio=False)  
            # ssd要求输入必须300*300，所以keep_ratio必须False，否则可能导致图片变小输出最后一层size计算为负
        img = torch.tensor(img).to(self.device).unsqueeze(0) 
        
        # 4. 数据包准备
        img_meta = [dict(ori_shape=ori_shape,
                         img_shape=img_shape,
                         pad_shape=pad_shape,
                         scale_factor = scale_factor,
                         flip=False)]
    
        data = dict(img=[img], img_meta=[img_meta])
        return data
    
    def run_single(self, model, img, data, by_plt=True):
        """对单张图片计算结果""" 
        with torch.no_grad():
            result = model(**data, return_loss=False, rescale=True)  # (20,)->(n,5)or(0,5)->(xmin,ymin,xmax,ymax,score)
            
            # 提取labels
            
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) 
                        for i, bbox in enumerate(result)]    # [(m1,), (m2,)...]
            labels = np.concatenate(labels)  # (m,)
            bboxes = np.vstack(result)       # (m,5)
            scores = bboxes[:,-1]
            
            if by_plt:
                vis_bbox(img.copy(), bboxes, label=labels, score=scores, 
                         score_thr=0.2, label_names=self.class_names,
                         instance_colors=None, alpha=1., linewidth=1.5, ax=None)
    
    def run(self, img_path):
        """根据需要继承该Tester然后修改run()即可"""
        # cfg, model
        cfg, model = self.init_cfg_model()
        # img
        img = cv2.imread(img_path)
        data = self.preprocess_img(cfg, img)
        # run
        self.run_single(model, img, data)
        

class TestImg(Tester):
    
    def __init__(self, config_file, model_class, weights_path, 
                 dataset_name='voc', device = 'cuda:0'):
        super().__init__(config_file, model_class, weights_path, 
                 dataset_name, device = 'cuda:0')
        

class TestCam(Tester):
    
    def __init__(self, config_file, model_class, weights_path, 
                 dataset_name='voc', device = 'cuda:0'):       
        super().__init__(config_file, model_class, weights_path, 
                 dataset_name, device = 'cuda:0')
        
    def run(self):
        """"""
        # cfg, model
        cfg, model = self.init_cfg_model()
        #
        cam_id = 0
        capture = cv2.VideoCapture(cam_id)
        while True:
            ret, img = capture.read()
            if not ret:
                cv2.destroyAllWindows()
                capture.release()
                break
            data = self.preprocess_img(cfg, img)
            
            self.run_single(model, img, data)
            result_img = []
            
    

if __name__ == "__main__":     
    
    test_this_img = True
    
    if test_this_img:
        img_path = './data/misc/test.jpg'    
        config_file = './config/cfg_ssd300_vgg16_voc.py'
        model_class = OneStageDetector
        weights_path = './weights/myssd/weight_4imgspergpu/epoch_24.pth'
        dataset_name = 'voc'

        test_img = TestImg(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
        test_img.run(img_path)
#        test_cam = TestCam(config_file, model_class, weights_path, dataset_name, device = 'cuda:0')
#        test_cam.run()
