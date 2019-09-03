#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:30:35 2019

@author: ubuntu
"""

log_level = 'INFO'               # 用于定义输出内容：INFO为基础输出内容，DEBUG为完整输出内容
gpus = 0
task = 'classifier'              # 用于定义任务类型：classifier, detector

model = dict(                    # model是必须要有的参数，用来表示主检测器集成模型或者单分类器模型
        type='alexnet8',          
        params=dict(
                n_classes=10))

img_transform = dict(
        mean=[0.49139968 0.48215841 0.44653091], 
        std=[0.06052839 0.06112497 0.06764512], 
        to_rgb=False, 
        to_tensor=True, 
        to_chw=False, 
        flip=None, 
        scale=None, 
        keep_ratio=None
        )

dataset = dict(
        type='cifar10',
        repeat=0,
        params=dict(
                root_path='../dataset/source/cifar10/', 
                data_type='train',
                norm=None, 
                label_transform_dict=None, 
                one_hot=None, 
                binary=None, 
                shuffle=None))


imgs_per_core = 4                  # 如果是gpu, 则core代表gpu，否则core代表cpu
workers_per_core = 2

dataloader = dict(
        params=dict(
                shuffle=False,
                batch_size=gpus * imgs_per_core if gpus>0 else imgs_per_core,
                num_workers=gpus * workers_per_core if gpus>0 else imgs_per_core,
                pin_memory=False,   # 数据送入GPU进行加速(默认False)
                drop_last=False))   # 最后一个batch
# 待增加学习率调整模块
optimizer = dict(
        type='sgd',
        params=dict(
                lr=2e-4, 
                momentum=0.9, 
                weight_decay=5e-4))

loss = dict(
        type='cross_entropy',
        params=dict(
                ))