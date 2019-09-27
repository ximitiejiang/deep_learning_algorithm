#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:35:56 2019

@author: ubuntu
"""

from utils.transform import get_dataset_norm_params, normalize
from dataset.cifar_dataset import Cifar10Dataset

"""
计算一个数据集的均值和方差：输入图片需基于chw，rgb格式。
    实例：参考mmcv中cifar10的数据mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]
    以上是先归一化到[0-1]之后再求得均值和方差，本方法所求结果完全一致，区别在于本方法是BGR顺序
"""

# 直接求标准化的mean, std
dataset = Cifar10Dataset()
mean, std = get_dataset_norm_params(dataset)  # 要求数据集为hwc, bgr
print("dataset mean: ", mean)  # [113.86538318 122.95039414 125.30691805]
print("dataset std: ", std)    # [51.22018275 50.82543151 51.56153984]


# 先归一化到[0,1]然后再求标准化的mean, std
dataset = Cifar10Dataset(img_transform = normalize)
mean, std = get_dataset_norm_params(dataset)
print("dataset mean: ", mean)  # [0.44653091 0.48215841 0.49139968]
print("dataset std: ", std)    # [0.20086346 0.19931542 0.20220212]