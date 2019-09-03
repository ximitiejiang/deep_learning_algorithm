#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:35:56 2019

@author: ubuntu
"""

from utils.transformer import get_dataset_norm_params
from dataset.cifar_dataset import Cifar10Dataset
from utils.transformer import normalize
"""
计算一个数据集的均值和方差：输入图片需基于chw，rgb格式。
    实例：参考mmcv中cifar10的数据mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]
    以上是先归一化到[0-1]之后再求得均值和方差，本方法所求结果跟该mmcv在std上稍有差异，待澄清
"""
dataset = Cifar10Dataset(img_transform = normalize)
mean, std = get_dataset_norm_params(dataset)
print("dataset mean: ", mean)
print("dataset std: ", std)
