#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:30:37 2019

@author: ubuntu
"""
import cv2
import numpy as np
import torch
from utils.config import Config
from utils.grad_cam import GradCam
from dataset.voc_dataset import VOCDataset
from dataset.utils import get_dataset

class TestOnVOC():
    
    def __init__(self, model, img_path, use_cuda=True):
        self.model = model
        self.img = cv2.imread(img_path, 1)
        self.use_cuda = use_cuda
        self.cfg_path = '../config/cfg_ssd300_vgg16_voc.py'
        self.cfg = Config.fromfile(self.cfg_path)
        self.cfg.data_root = '../data/VOCdevkit/'
        # train
        self.cfg.data.train.dataset.ann_file = [
                self.cfg.data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                self.cfg.data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ]
        self.cfg.data.train.dataset.img_prefix = [self.cfg.data_root + 'VOC2007/', self.cfg.data_root + 'VOC2012/']
        # val
        self.cfg.data.val.ann_file = self.cfg.data_root + 'VOC2007/ImageSets/Main/test.txt'
        self.cfg.data.val.img_prefix = self.cfg.data_root + 'VOC2007/'
        # test
        self.cfg.data.test.ann_file = self.cfg.data_root + 'VOC2007/ImageSets/Main/test.txt'
        self.cfg.data.test.img_prefix = self.cfg.data_root + 'VOC2007/'
        
        self.dataset = get_dataset(self.cfg.data.train, VOCDataset)
        
        # img preprocess
        self.img = np.float32(cv2.resize(self.img, (224, 224))) / 255  # 变更尺寸，归一化
        self.img = self.preprocess_image()  # (1, c, h, w)
        
    def preprocess_image(self):
    	means=[0.485, 0.456, 0.406]
    	stds=[0.229, 0.224, 0.225]
    
    	preprocessed_img = self.img.copy()[: , :, ::-1]
    	for i in range(3):
    		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
    		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    	preprocessed_img = \
    		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    	preprocessed_img = torch.from_numpy(preprocessed_img)
    	preprocessed_img.unsqueeze_(0)
    	self.img = torch.tensor(preprocessed_img, requires_grad = True)
            
        
    def showcam(self, class_index):
        use_cuda = self.use_cuda

    	# Can work with any model, but it assumes that the model has a 
    	# feature method, and a classifier method,
    	# as in the VGG models in torchvision.
        """stpe1: 创建grad_cam对象，传入model/layer, 但要确保model里边有model.feature, model.classifier"""
        grad_cam = GradCam(model = self.model, \
                           target_layer_names = ["35"], use_cuda=use_cuda)
        
        """step2: 图片预处理，"""

    
    	# If None, returns the map for the highest scoring category.
    	# Otherwise, targets the requested index.
        class_index = [1,3]   # 输入None则表示希望返回得分最高的类别，也可输入类的index，比如[2,5,12]就是希望输出第2,5,12类的特征映射情况
        mask = grad_cam(input, class_index)    
        
        # show_cam_on_image(img, mask)
        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(self.img)
        cam = cam / np.max(cam)
        cv2.imwrite("cam.jpg", np.uint8(255 * cam))


if __name__ == '__main__':
    from torchvision import models
    img_path = '../data/misc/088462.jpg'
    model = models.vgg19(pretrained=True)
    
    tov = TestOnVOC(model, img_path, use_cuda=True)
    tov.showcam(class_index=None)