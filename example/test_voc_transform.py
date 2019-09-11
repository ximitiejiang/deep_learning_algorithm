#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:18:42 2019

@author: ubuntu
"""
import cv2
import numpy as np
from utils.prepare_training import get_config, get_logger, get_dataset, get_dataloader 
from utils.prepare_training import get_model, get_optimizer, get_lr_processor, get_loss_fn
from utils.transform import transform_inv, imdenormalize

transform = dict(
        img_params=dict(
                mean=[123.675, 116.28, 103.53],  # 通常要基于BGR顺序的mean/std，但这里mmdetection是用RGB顺序的mean/std
                std=[1., 1., 1.],
                to_rgb=True,    # bgr to rgb
                to_tensor=True, # numpy to tensor 
                to_chw=True,    # hwc to chw
                flip_ratio=0.5,
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

trainloader = dict(
        params=dict(
                shuffle=True,
                batch_size=16,
                num_workers=1,
                pin_memory=False,   # 数据送入GPU进行加速(默认False)
                drop_last=False,
                collate_fn='dict_collate',    # 'default_collate','multi_collate', 'dict_collate'
                sampler=None))



# %%
"""验证思路，逐步增加变换内容，一项一项进行检查
"""

if __name__ == "__main__":

#    cfg_path = '../model/cfg_detector_ssdvgg16_voc.py'
#    cfg = get_config(cfg_path)
    
    trainset = get_dataset(trainset, transform)
    dataloader = get_dataloader(trainset, trainloader)
    class_names = trainset.CLASSES
    # 检查图片经transform后恢复的效果
    data = trainset[12]
    img = data['img']
    bboxes = data['gt_bboxes']
    labels = data['gt_labels']
    meta = data['img_meta']
    print(meta)
    # 最简应用
#    img = transform_inv(img, mean=transform['img_params']['mean'], std=transform['img_params']['std'], show=True)
    # 复杂应用
    img, bboxes, labels = transform_inv(img, bboxes, labels, 
                                        transform['img_params']['mean'], 
                                        transform['img_params']['std'],
                                        class_names=class_names,
                                        show=True)
    
    # 检查图片经dataloader后恢复的效果
    data_batch = next(iter(dataloader))
    img = data_batch['img']
    
