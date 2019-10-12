#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:30:35 2019

@author: ubuntu
"""
task = 'classifier'              # 用于定义任务类型：classifier, detector, regressor
gpus = 1
parallel = False
distribute = False                       
n_epochs = 4
imgs_per_core = 32               # 如果是gpu, 则core代表gpu，否则core代表cpu(等效于batch_size)
workers_per_core = 0
save_checkpoint_interval = 10     # 每多少个epoch保存一次epoch
work_dir = '/home/ubuntu/mytrain/resnet_cifar10/'
resume_from = None               # 恢复到前面指定的设备
load_from = None
load_device = 'cuda'              # 额外定义用于评估预测的设备: ['cpu', 'cuda']，可在cpu预测

lr = 0.001

lr_processor = dict(
        type='list',
        params = dict(
                step=[6, 8],       # 代表第2个(从1开始算)
                lr = [0.001, 0.0001],
                warmup_type='linear',
                warmup_iters=500,
                warmup_ratio=1./3))

logger = dict(
                log_level='INFO',
                log_dir=work_dir,
                interval=100)
model = dict(
        type='classifier')

backbone = dict(                    # model是必须要有的参数，用来表示主检测器集成模型或者单分类器模型
        type='resnet',       
        params=dict(
                depth=18,
                pretrained='/home/ubuntu/MyWeights/resnet18-5c106cde.pth',
                out_indices=[],    # 为空则输出最后的classifier
                classify_classes=10))

transform = dict(
        img_params=dict(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225],
                norm=True,
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=None,
                keep_ratio=None),
        label_params=dict(
                to_tensor=True,
                to_onehot=None),
        bbox_params=None)

transform_val = dict(
        img_params=dict(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225],
                norm=True,
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=None,
                keep_ratio=None),
        label_params=dict(
                to_tensor=True,
                to_onehot=None))

data_root_path='/home/ubuntu/MyDatasets/cifar-10-batches-py/'  # 统一一个data_root_path变量，便于书写，也便于check
trainset = dict(
        type='cifar10',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                data_type='train'))
valset = dict(
        type='cifar10',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                data_type='test'))

trainloader = dict(
        params=dict(
                shuffle=True,
                batch_size=gpus * imgs_per_core if gpus>0 else imgs_per_core,
                num_workers=gpus * workers_per_core if gpus>0 else imgs_per_core,
                pin_memory=False,   # 数据送入GPU进行加速(默认False)
                drop_last=False,
                collate_fn='dict_collate',    # 'default_collate','multi_collate', 'dict_collate'
                sampler=None))
valloader = dict(        
        params=dict(
                shuffle=False,
                batch_size=gpus * imgs_per_core if gpus>0 else imgs_per_core,
                num_workers=gpus * workers_per_core if gpus>0 else imgs_per_core,
                pin_memory=False,   # 数据送入GPU进行加速(默认False)
                drop_last=False,
                collate_fn='dict_collate',    # 'default_collate','multi_collate', 'dict_collate'
                sampler=None))

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
