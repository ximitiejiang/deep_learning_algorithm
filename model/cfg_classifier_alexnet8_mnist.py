#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:30:35 2019

@author: ubuntu
"""
"""--------------------------------------------------------------------------
基础设置选择
1.optimizer:
    - sgd: 
    - 
    
   --------------------------------------------------------------------------
"""

task = 'classifier'              # 用于定义任务类型：classifier, detector, regressor
gpus = 1
parallel = False
distribute = False                       
n_epochs = 5
imgs_per_core = 32               # 如果是gpu, 则core代表gpu，否则core代表cpu(等效于batch_size)
workers_per_core = 2
save_checkpoint_interval = 1     # 每多少个epoch保存一次epoch
work_dir = '/home/ubuntu/mytrain/alexnet8/'
resume_from = None               # 恢复到前面指定的设备
load_from = '/home/ubuntu/mytrain/alexnet8/epoch_5.pth'
load_device = 'cuda'              # 额外定用于评估预测的设备: ['cpu', 'cuda']，可在cpu预测

lr = 0.01

lr_processor = dict(
        type='list',
        params = dict(
                step=[2, 4],       # 代表第2个(从1开始算，等效于n_epochs的算法)
                lr = [0.005, 0.002],
                gamma=0.1,
                warmup_type='linear',
                warmup_iters=500,
                warmup_ratio=1./3))

logger = dict(
                log_level='INFO',
                log_dir=work_dir,
                interval=50)

model = dict(                    # model是必须要有的参数，用来表示主检测器集成模型或者单分类器模型
        type='alexnet8',          
        params=dict(
                n_classes=10))

transform = dict(
        img_params=dict(
                mean=[113.86538318, 122.95039414, 125.30691805],  # 基于BGR顺序
                std=[51.22018275, 50.82543151, 51.56153984],
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip=None,
                scale=None,
                keep_ratio=None),
        label_params=dict(
                to_tensor=True,
                to_onehot=None),
        bbox_params=None)

transform_val = dict(
        img_params=dict(
                mean=[113.86538318, 122.95039414, 125.30691805],  # 基于BGR顺序
                std=[51.22018275, 50.82543151, 51.56153984],
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip=None,
                scale=None,
                keep_ratio=None),
        label_params=dict(
                to_tensor=True,
                to_onehot=None))

trainset = dict(
        type='cifar10',
        repeat=0,
        params=dict(
                root_path='/home/ubuntu/MyDatasets/cifar-10-batches-py/', 
                data_type='train'))
valset = dict(
        type='cifar10',
        repeat=0,
        params=dict(
                root_path='/home/ubuntu/MyDatasets/cifar-10-batches-py/', 
                data_type='test'))

trainloader = dict(
        params=dict(
                shuffle=True,
                batch_size=gpus * imgs_per_core if gpus>0 else imgs_per_core,
                num_workers=gpus * workers_per_core if gpus>0 else imgs_per_core,
                pin_memory=False,   # 数据送入GPU进行加速(默认False)
                drop_last=False,
                collate_fn=None,    # 'multi_collate'
                sampler=None))
valloader = dict(        
        params=dict(
                shuffle=False,
                batch_size=gpus * imgs_per_core if gpus>0 else imgs_per_core,
                num_workers=gpus * workers_per_core if gpus>0 else imgs_per_core,
                pin_memory=False,   # 数据送入GPU进行加速(默认False)
                drop_last=False,
                collate_fn=None, # 'multi_collate'
                sampler=None))   # 最后一个batch
# 待增加学习率调整模块
optimizer = dict(
        type='sgd',
        params=dict(
                lr=lr, 
                momentum=0.9, 
                weight_decay=5e-4))

loss_clf = dict(
        type='cross_entropy',
        params=dict(
                reduction='mean'
                ))

#loss_reg = dict(
#        type='smooth_l1',
#        params=dict(
#                reduction='mean'
#                ))
