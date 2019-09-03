#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:30:35 2019

@author: ubuntu
"""

log_level = 'INFO'               # 用于定义输出内容：INFO为基础输出内容，DEBUG为完整输出内容
gpus = 0
task = 'classifier'              # 用于定义任务类型：classifier, detector

model = dict(
        type='alexnet8',          # 主模型(model)和子模型(backbone,neck,head)
        params=dict(
                n_classes=10))

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