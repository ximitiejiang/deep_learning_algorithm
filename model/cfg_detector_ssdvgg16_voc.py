#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:31:23 2019

@author: ubuntu
"""
log_level = 'INFO'  # 用于定义输出内容：INFO为基础输出内容，DEBUG为完整输出内容
gpus = 1

model = dict(
        type='one_stage_detector')
        
backbone=dict(
        type='alexnet8',
        params=dict(
                num_classes=10))

header=dict(
        type='ssd_head',
        params=dict(
                input_size=300,
                num_classes=21,
                in_channels=(512, 1024, 512, 256, 256, 256),
                num_anchors=(4, 6, 6, 6, 4, 4),
                anchor_strides=(8, 16, 32, 64, 100, 300),
                target_means=(.0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2)))

dataset = dict(
        type='Cifar10Dataset',
        repeat=5,
        params=dict(
                root_path='../dataset/source/cifar10/', 
                data_type='train',
                norm=None, 
                label_transform_dict=None, 
                one_hot=None, 
                binary=None, 
                shuffle=None))

imgs_per_gpu = 4
workers_per_gpu = 2

dataloader = dict(
        params=dict(
                shuffle=False,
                batch_size=gpus * imgs_per_gpu,
                num_workers=gpus * workers_per_gpu, 
                pin_memory=False,   # 数据送入GPU进行加速(默认False)
                drop_last=False))   # 最后一个batch

optimizer = dict(
        type = 'SGD'
        params = dict(
                ))