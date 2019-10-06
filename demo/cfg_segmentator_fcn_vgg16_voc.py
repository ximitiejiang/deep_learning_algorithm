#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:31:23 2019

@author: ubuntu
"""
task = 'segmentator'
gpus = 1
parallel = False
distribute = False                       
n_epochs = 10
imgs_per_core = 4                 # 如果是gpu, 则core代表gpu，否则core代表cpu(等效于batch_size)
workers_per_core = 2
save_checkpoint_interval = 2     # 每多少个epoch保存一次epoch
work_dir = '/home/ubuntu/mytrain/fcn_vgg_voc/'
resume_from = None                # 恢复到前面指定的设备
load_from = None
load_device = 'cuda'              # 额外定义用于评估预测的设备: ['cpu', 'cuda']，可在cpu预测

lr = 0.001

lr_processor = dict(
        type='list',
        params = dict(
                step=[5, 8],       # 代表第2个(从1开始算)
                lr = [0.0005, 0.0001],
                warmup_type='linear',
                warmup_iters=500,
                warmup_ratio=1./3))

logger = dict(
                log_level='INFO',
                log_dir=work_dir,
                interval=1)

model = dict(
        type='segmentator')
        
backbone = dict(
        type='fcn_vgg16',
        params=dict(
                depth=16,
                pretrained= '/home/ubuntu/MyWeights/vgg16-397923af.pth',   # pytorch的模型
                out_indices=(16, 23, 30)))

head = dict(
        type='fcn8s_head',
        params=dict(
                in_channels=(256, 512, 512),
                num_classes=21,
                featmap_sizes=(60, 30, 15),
                out_size=480,
                out_layer=0,
                upsample_method='interpolate'))

transform = dict(
        img_params=dict(
                mean=[0.485, 0.456, 0.406],  # 基于RGB顺序: 基于归一化之后进行，所以norm要等于True
                std=[0.229, 0.224, 0.225],
                norm=True,     # 归一化img/255
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=(480, 480),  # 选择300的小尺寸
                size_divisor=None,
                keep_ratio=False),  # ssd需要统一到方形300,300，不能按比例
        label_params=None,
        bbox_params=None,
        aug_params=None,
        mask_params=None,
        seg_params=dict(
                to_tensor=True,
                scale=(480,480),
                keep_ratio=False,
                size_divisor=None,
                seg_scale_factor=None))
        
transform_val = dict(
        img_params=dict(
                mean=[0.485, 0.456, 0.406],  # 基于RGB顺序: 基于归一化之后进行，所以norm要等于True
                std=[0.229, 0.224, 0.225],
                norm=True,     # 归一化img/255
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=None,
                scale=(480, 480),  # [w,h]
                size_divisor=None,
                keep_ratio=False),
        label_params=None,
        bbox_params=None,
        seg_params=dict(
                to_tensor=True,
                scale=(480,480),
                keep_ratio=False,
                size_divisor=None,
                seg_scale_factor=None))

data_root_path='/home/ubuntu/MyDatasets/voc/VOCdevkit/'
trainset = dict(
        type='voc',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'VOC2007/ImageSets/Segmentation/trainval.txt',  # 分为train.txt, val.txt, trainval.txt, test.txt
                          data_root_path + 'VOC2012/ImageSets/Segmentation/trainval.txt'], # 注意，分割任务的ann file需要采用纯分割的ann file
                img_prefix=[data_root_path + 'VOC2007/',
                          data_root_path + 'VOC2012/'],
                seg_prefix='SegmentationClass/',     # 识别是用SegmentationClass还是用SegmentationObject
                ))
valset = dict(
        type='voc',
        repeat=0,
        params=dict(
                root_path=data_root_path, 
                ann_file=[data_root_path + 'VOC2007/ImageSets/Segmentation/test.txt'],   #注意只有2007版本有test.txt，到2012版取消了。
                img_prefix=[data_root_path + 'VOC2007/'],  
                seg_prefix='SegmentationClass/',                       
                ))

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
                batch_size=1 * 1,  # 做验证时需要让batch_size=1???
                num_workers=gpus * workers_per_core if gpus>0 else imgs_per_core,
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

