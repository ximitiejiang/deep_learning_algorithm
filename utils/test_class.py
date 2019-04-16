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
from dataset.utils import vis_bbox, opencv_vis_bbox
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
    
    def preprocess_data(self, cfg, img, transformer=None):
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
    
    def run_single(self, model, img, data, show=False, saveto=None):
        """对单张图片计算结果""" 
        with torch.no_grad():
            result = model(**data, return_loss=False, rescale=True)  # (20,)->(n,5)or(0,5)->(xmin,ymin,xmax,ymax,score)         
        # 提取labels
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) 
                    for i, bbox in enumerate(result)]    # [(m1,), (m2,)...]
        labels = np.concatenate(labels)  # (m,)
        bboxes = np.vstack(result)       # (m,5)
        scores = bboxes[:,-1]
        
        all_results = [bboxes]
        all_results.append(labels)
        all_results.append(scores)
        
        if show:
            vis_bbox(
                img.copy(), *all_results, score_thr=0.2, class_names=self.class_names, 
                instance_colors=None, alpha=1., linewidth=1.5, ax=None, saveto=saveto)
            # opencv版本的显示效果不太好，用matplotlib版本的显示文字较好
#            opencv_vis_bbox(
#                img.copy(), *all_results, score_thr=0.2, class_names=self.class_names, 
#                instance_colors=None, thickness=1, font_scale=0.5,
#                show=True, win_name='test_pic', wait_time=1, out_file=None)
            
        return all_results
            
    def run(self, img_path):
        raise NotImplementedError('run() function not implemented!')
        

class TestImg(Tester):
    
    def __init__(self, config_file, model_class, weights_path, 
                 dataset_name='voc', device = 'cuda:0'):
        super().__init__(config_file, model_class, weights_path, 
                 dataset_name, device = 'cuda:0')
    
    def run(self, img_path):
        # cfg, model
        cfg, model = self.init_cfg_model()
        # img
        img = cv2.imread(img_path)
        data = self.preprocess_data(cfg, img)
        # run
        _ = self.run_single(model, img, data, show=True, saveto='result.jpg')
        

class TestVideo(Tester):
    """用于视频或者摄像头的检测"""
    def __init__(self, config_file, model_class, weights_path, 
                 dataset_name='voc', device = 'cuda:0'):       
        super().__init__(config_file, model_class, weights_path, 
                 dataset_name, device = 'cuda:0')        
        
    def run(self, source):
        """"""
        # source can be int(as cam_id) or str(as video path)
        if isinstance(source, int):
            cam_id = source
            capture = cv2.VideoCapture(cam_id)
        elif isinstance(source, str):
            capture = cv2.VideoCapture(source)
        assert capture.isOpened(), 'Cannot capture source'
        
        cfg, model = self.init_cfg_model()
        
        while True:
            ret, img = capture.read()
            if not ret:
                cv2.destroyAllWindows()
                capture.release()
                break
            data = self.preprocess_data(cfg, img)
            
            all_results = self.run_single(model, img, data, show=False, saveto=None)
            opencv_vis_bbox(img.copy(), *all_results, score_thr=0.5, class_names=self.class_names, 
                            instance_colors=None, thickness=1, font_scale=0.5,
                            show=True, win_name='cam', wait_time=0, out_file=None)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                capture.release()
                break
