#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:31:23 2019

@author: ubuntu
"""

gpus = [0]
parallel = False
distribute = False                       
n_epochs = 100
imgs_per_core = 4                 # 如果是gpu, 则core代表gpu，否则core代表cpu(等效于batch_size)
workers_per_core = 2
save_checkpoint_interval = 10     # 每多少个epoch保存一次epoch
work_dir = '/home/ubuntu/mytrain/ssd_vgg_voc/'
resume_from = None                # 恢复到前面指定的设备
load_from = None
load_device = 'cuda'              # 额外定义用于评估预测的设备: ['cpu', 'cuda']，可在cpu预测
lr = 0.001
img_size = (300, 300)

lr_processor = dict(
        type='list',
        params = dict(
                step=[10, 20],       # 代表第2个(从1开始算)
                lr = [0.0005, 0.0001],
                warmup_type='linear',
                warmup_iters=500,
                warmup_ratio=1./3))

logger = dict(
                log_level='INFO',
                log_dir=work_dir,
                interval=10)

model = dict(
        type='one_stage_detector')
        
backbone = dict(
        type='ssd_vgg16',
        params=dict(
                pretrained= '/home/ubuntu/MyWeights/pytorch/vgg16_caffe-292e1171.pth',   # 这是caffe的模型，对应mean=[123.675, 116.28, 103.53], std=[1, 1, 1],  另外的pytorch的模型pretrained='/home/ubuntu/.torch/models/vgg16-397923af.pth', 对应mean, std需要先归一化再标准化
                out_feature_indices=(22,34),
                extra_out_feature_indices=(1,3,5,7),
                l2_norm_scale=20.))

head = dict(
        type='ssd_head',
        params=dict(
                input_size=img_size,
                num_classes=21,
                in_channels=(512, 1024, 512, 256, 256, 256),
                num_anchors=(4, 6, 6, 6, 4, 4),
                anchor_size_ratio_range=(0.2, 0.9),
                anchor_ratios = ([2],[2, 3],[2, 3],[2, 3],[2],[2]),
                anchor_strides=(8, 16, 32, 64, 100, 300),
                target_means=(.0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
                neg_pos_ratio=3))

assigner = dict(
        type='max_iou_assigner',
        params=dict(
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.))

sampler = dict(
        type='posudo_sampler',
        params=dict(
                ))

neg_pos_ratio = 3  
nms = dict(
        type='nms',
        score_thr=0.02,
        max_per_img=200,
        params=dict(
                iou_thr=0.5)) # nms过滤iou阈值

transform = dict(
        img_params=dict(
                mean=[123.675, 116.28, 103.53],  # 基于RGB顺序: 由于采用caffe的backbone模型，所以图片归一化参数基于caffe
                std=[1, 1, 1],
                norm=False,     # 归一化img/255
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=img_size,  # 选择300的小尺寸
                size_divisor=None,
                keep_ratio=False),  # ssd需要统一到方形300,300，不能按比例
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
                scale=img_size,  # [w,h]
                size_divisor=None,
                keep_ratio=False),
        label_params=dict(
                to_tensor=True,
                to_onehot=None),
        bbox_params=dict(
                to_tensor=True
                ))

data_root_path='/home/ubuntu/MyDatasets0/voc/VOCdevkit/'  #
trainset = dict(
        type='voc',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'VOC2007/ImageSets/Main/trainval.txt',
                          data_root_path + 'VOC2012/ImageSets/Main/trainval.txt'], #分为train.txt, val.txt, trainval.txt, test.txt
                img_prefix=[data_root_path + 'VOC2007/',
                            data_root_path + 'VOC2012/'],
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
                batch_size=imgs_per_core,
                num_workers=workers_per_core,
                pin_memory=False,   # 数据送入GPU进行加速(默认False)
                drop_last=False,
                collate_fn='dict_collate',    # 'default_collate','multi_collate', 'dict_collate'
                sampler=None))

valloader = dict(        
        params=dict(
                shuffle=False,
                batch_size=1 * 1,  # 做验证时需要让batch_size=1???
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

