#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:31:23 2019

@author: ubuntu
"""
task = 'detector'                # 用于定义任务类型：classifier, detector, regressor
gpus = 1
parallel = False
distribute = False                       
n_epochs = 1
imgs_per_core = 64               # 如果是gpu, 则core代表gpu，否则core代表cpu(等效于batch_size)
workers_per_core = 2
save_checkpoint_interval = 2     # 每多少个epoch保存一次epoch
work_dir = '/home/ubuntu/mytrain/ssd_vgg_voc/'
resume_from = None               # 恢复到前面指定的设备
load_from = None
load_device = 'cuda'             # 额外定义用于评估预测的设备: ['cpu', 'cuda']，可在cpu预测

lr = 0.01

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
        type='one_stage_detector')
        
backbone=dict(
        type='ssdvgg16',
        params=dict(
                pretrained='',
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

transform = dict(
        img_params=dict(
                mean=[123.675, 116.28, 103.53], std=[1, 1, 1],  # 基于BGR顺序: 由于采用caffe的backbone模型，所以图片归一化参数基于caffe
                std=[1, 1, 1],
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=[512, 512],
                size_divisor=None,
                keep_ratio=True),
        label_params=dict(
                to_tensor=True,
                to_onehot=None),
        bbox_params=dict(
                to_tensor=True
                ),
        aug_params=None)

transform_val = dict(
        img_params=dict(
                mean=[113.86538318, 122.95039414, 125.30691805],  # 基于BGR顺序
                std=[51.22018275, 50.82543151, 51.56153984],
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=[512, 512],  # [w,h]
                size_divisor=None,
                keep_ratio=True),
        label_params=dict(
                to_tensor=True,
                to_onehot=None),
        bbox_params=None)

data_root_path='/home/ubuntu/MyDatasets/voc/VOCdevkit/'
trainset = dict(
        type='voc',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'VOC2007/ImageSets/Main/trainval.txt',
                          data_root_path + 'VOC2012/ImageSets/Main/trainval.txt'], #分为train.txt, val.txt, trainval.txt, test.txt
                subset_path=[data_root_path + 'VOC2007/',
                          data_root_path + 'VOC2012/'],
                data_type='train'))
valset = dict(
        type='voc',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'VOC2007/ImageSets/Main/test.txt'],
                subset_path=[data_root_path + 'VOC2007/'],                         #注意只有2007版本有test.txt，到2012版取消了。
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

loss_reg = dict(
        type='smooth_l1',
        params=dict(
                reduction='mean'
                ))
