#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:31:23 2019

@author: ubuntu
"""

gpus = [0]
parallel = False
distribute = False                       
n_epochs = 50
imgs_per_core = 32                 # 如果是gpu, 则core代表gpu，否则core代表cpu(等效于batch_size)
workers_per_core = 2
save_checkpoint_interval = 5     # 每多少个epoch保存一次epoch
work_dir = '/home/ubuntu/mytrain/retinaface_widerface/'
resume_from = None                # 恢复到前面指定的设备
load_from = None
load_device = 'cuda'              # 额外定义用于评估预测的设备: ['cpu', 'cuda']，可在cpu预测
lr = 0.001
img_size = (640, 640)

lr_processor = dict(
        type='list',
        params = dict(
                step=[20, 40, 60],       # 代表第2个(从1开始算)
                lr = [0.0005, 0.0001, 0.00005],
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
        type='mobilenet_v1',
        params=dict(
                pretrained= '/home/ubuntu/MyWeights/retinaface_backbone/mobilenetV1X0.25_pretrain.tar',   # 这是caffe的模型，对应mean=[123.675, 116.28, 103.53], std=[1, 1, 1],  另外的pytorch的模型pretrained='/home/ubuntu/.torch/models/vgg16-397923af.pth', 对应mean, std需要先归一化再标准化
                out_stages=(0,1,2)))

neck = dict(
        type='fpnssh',
        params=dict(
                in_channels=(64,128,256),
                out_channels=64,
                use_levels=(0,1,2),
                num_outs=3))

head = dict(
        type='retinaface_head',
        params=dict(
                input_size=img_size,
                in_channels=(64, 64, 64),
                num_classes=2,
                num_landmarks=10,
                base_sizes=(16, 64, 256),
                strides=(8,16,32),
                scales=(1,2),
                ratios=(1),
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

nms = dict(
        type='nms',
        score_thr=0.02,
        max_per_img=200,
        params=dict(
                iou_thr=0.5)) # nms过滤iou阈值

transform = dict(
        aug_params=dict(
                crop_size=img_size),
        img_params=dict(
                mean=[123.675, 116.28, 103.53],  # 基于RGB顺序
                std=[1, 1, 1],
                norm=False,     # 归一化img/255
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=img_size,
                size_divisor=None,
                keep_ratio=False),  # 需要正好640x640，但因前面有aug transform，所以不会比例变形
        label_params=dict(
                to_tensor=True,
                to_onehot=None),
        bbox_params=dict(
                to_tensor=True
                ),
        landmark_params=dict(
                to_tensor=True))

transform_val = dict(
        img_params=dict(
                mean=[123.675, 116.28, 103.53],  # 基于RGB顺序: 该数据来自retinaface的mean, std没有则为1，且明确为rgb顺序
                std=[1, 1, 1],
                norm=False,
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=None,     # 验证时不缩放图片，防止比例变形
                size_divisor=32, # 对于既有下采样又有上采样的图片，最后增加size divisor，否则可能因为下采样floor导致上采样无法merge
                keep_ratio=False),
        label_params=dict(
                to_tensor=True,
                to_onehot=None),
        bbox_params=dict(
                to_tensor=True
                ),
        landmark_params=dict(
                to_tensor=True))

data_root_path='/home/ubuntu/MyDatasets/WIDERFace/'
trainset = dict(
        type='widerface',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'WIDER_train/train.txt'], #分为train.txt, val.txt, trainval.txt, test.txt
                img_prefix=[data_root_path + 'WIDER_train/'],
                landmark_file=[data_root_path + 'WIDER_train/label.txt'],  # 采用带关键点的数据
                ))
valset = dict(
        type='widerface',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'WIDER_val/val.txt'],   
                img_prefix=[data_root_path + 'WIDER_val/'],
                landmark_file=[data_root_path + 'WIDER_val/label.txt'],                         
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
