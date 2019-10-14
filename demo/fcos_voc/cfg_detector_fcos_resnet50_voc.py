#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:31:23 2019

@author: ubuntu
"""

gpus = 1
parallel = False
distribute = False                       
n_epochs = 10
imgs_per_core = 4               # 如果是gpu, 则core代表gpu，否则core代表cpu(等效于batch_size)
workers_per_core = 4
save_checkpoint_interval = 1     # 每多少个epoch保存一次epoch
work_dir = '/home/ubuntu/mytrain/fcos_resnet50_voc/'
resume_from = None               # 恢复到保存的设备
load_from = None
load_device = 'cuda'             # 额外定义用于评估预测的设备: ['cpu', 'cuda']，可在cpu预测

lr = 0.001

lr_processor = dict(
        type='list',
        params = dict(
                step=[4, 8],       
                lr = [0.0005, 0.0001],
                warmup_type='linear',
                warmup_iters=500,
                warmup_ratio=1./3))

logger = dict(
                log_level='INFO',
                log_dir=work_dir,
                interval=1)

model = dict(
        type='one_stage_detector')
        
backbone = dict(
        type='resnet',
        params=dict(
                depth=50,
                pretrained= '/home/ubuntu/MyWeights/resnet50-19c8e357.pth',   # 这是caffe的模型，对应mean=[123.675, 116.28, 103.53], std=[1, 1, 1],  另外的pytorch的模型pretrained='/home/ubuntu/.torch/models/vgg16-397923af.pth', 对应mean, std需要先归一化再标准化
                out_indices=(0, 1, 2, 3),
                strides=(1, 2, 2, 2)))

neck = dict(
        type='fpn',
        params=dict(
                in_channels=(256, 512, 1024, 2048),
                out_channels=256,
                use_levels=(1, 2, 3),
                num_outs=5,
                extra_convs_on_input=True
                ))

head = dict(
        type='fcos_head',
        params=dict(
                num_classes=21,
                in_channels=256,
                out_channels=256,
                num_convs=4,
                strides=(4,8,16,32,64),
                regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)),
                ))

nms = dict(
        type='nms',
        score_thr=0.02,
        max_per_img=200,
        params=dict(
                iou_thr=0.5)) # nms过滤iou阈值

transform = dict(
        img_params=dict(
                mean=[123.675, 116.28, 103.53],   # 注意要用caffe在voc的mean std
                std=[1, 1, 1],
                norm=False,     # 归一化img/255
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=(1333, 800),  # 选择300的小尺寸
                size_divisor=32,
                keep_ratio=True),  # ssd需要统一到方形300,300，不能按比例
        label_params=dict(
                to_tensor=True,
                to_onehot=None),
        bbox_params=dict(
                to_tensor=True
                ),
        aug_params=None)

transform_val = dict(
        img_params=dict(
                mean=[123.675, 116.28, 103.53],  # 基于RGB顺序: 由于采用caffe的backbone模型，所以图片归一化参数基于caffe
                std=[1, 1, 1],
                norm=False,
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=(1333, 800),  # [w,h]
                size_divisor=32,
                keep_ratio=True),
        label_params=dict(
                to_tensor=True,
                to_onehot=None),
        bbox_params=dict(
                to_tensor=True
                ))

data_root_path='/home/ubuntu/MyDatasets/voc/VOCdevkit/'
trainset = dict(
        type='voc',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'VOC2007/ImageSets/Main/trainval.txt'],
#                          data_root_path + 'VOC2012/ImageSets/Main/trainval.txt'], #分为train.txt, val.txt, trainval.txt, test.txt
                img_prefix=[data_root_path + 'VOC2007/'],
#                          data_root_path + 'VOC2012/'],
                ))
valset = dict(
        type='voc',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'VOC2007/ImageSets/Main/test.txt'],   #注意只有2007版本有test.txt，到2012版取消了。
                img_prefix=[data_root_path + 'VOC2007/'],                         
                ))

trainloader = dict(
        params=dict(
                shuffle=True,
                batch_size=imgs_per_core,         # 设置成单块GPU的size
                num_workers=workers_per_core,
                pin_memory=False,   # 数据送入GPU进行加速(默认False)
                drop_last=False,
                collate_fn='dict_collate',    # 'default', 'multi_collate', 'dict_collate'
                sampler=None))

valloader = dict(        
        params=dict(
                shuffle=False,
                batch_size=1,  # 做验证时需要让batch_size=1
                num_workers=workers_per_core,
                pin_memory=False,             # 数据送入GPU进行加速(默认False)
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
        beta=1.,
        params=dict(
                reduction='mean'
                ))
