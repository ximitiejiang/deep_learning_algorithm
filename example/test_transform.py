#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:09:14 2019

@author: ubuntu
"""
import matplotlib.pyplot as plt
from dataset.cifar_dataset import Cifar10Dataset
from utils.transformer import ImgTransform
"""显示区别
常规数据集出来的图片都是hwc,bgr格式
1. plt.imshow(), 支持hwc, rgb
2. cv2.imshow(), 支持hwc, bgr
"""

transform =dict(mean=[0.49139968 0.48215841 0.44653091],
                std=[0.06052839 0.06112497 0.06764512],
                to_rgb=False,    # 源数据集已是rgb
                to_tensor=True,  
                to_chw=False,    # 源数据集已是chw
                flip=None,
                scale=None,
                keep_ratio=None)
transforms = ImgTransform()

dataset = Cifar10Dataset()


img0, label0 = dataset[0] # hwc, bgr
img1, label1 = dataset[1]
img2, label2 = dataset[2]
img3, label3 = dataset[3]
imgs = [img0, img1, img2, img3]
labels = [label0, label1, label2, label3]
classes = [dataset.CLASSES[labels[i]] for i in range(4)]
plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i][..., [2,1,0]])
    plt.title(classes[i])