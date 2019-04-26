#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:50:59 2019

@author: ubuntu
"""
import pytest
import numpy as np
from dataset.utils import vis_bbox
from dataset.utils import get_dataset
from dataset.voc_dataset import VOCDataset
from dataset.coco_dataset import CocoDataset
from torch.utils.data import Dataloader

def test_voc_dataset():
    data_root = '../data/VOCdevkit/'  # 指代ssd目录下的data目录
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
    cfg_train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='VOCDataset',
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            img_scale=(300, 300),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 4)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
            resize_keep_ratio=False))
    
    trainset = get_dataset(cfg_train, VOCDataset)
    classes = trainset.CLASSES
    data = trainset[1120]  # dict('img', 'img_meta', )
    """已做的数据处理：rgb化，chw化，归一化，tensor化"""
    bbox = data['gt_bboxes'].data.numpy()
    label = data['gt_labels'].data.numpy()
    img = data['img'].data.numpy()     # 逆tensor
    img1 = img.transpose(1,2,0)   # 逆chw
    img2 = np.clip((img1 * img_norm_cfg['std'] + img_norm_cfg['mean']).astype(np.int32), 0, 255)  # 逆归一
#        plt.imshow(img2)
    vis_bbox(img2[...,[2,0,1]], bbox, label-1, label_names=classes)  # vis_bbox内部会bgr转rgb，所以这里要用bgr输入


def test_coco_dataset():
    
    data_root = '../data/coco/'
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
    cfg_train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='CocoDataset',
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_scale=(300, 300),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 4)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
            resize_keep_ratio=False))

    trainset = get_dataset(cfg_train, CocoDataset)
    classes = trainset.CLASSES
    data = trainset[0]
    """已做的数据处理：rgb化，chw化，归一化，tensor化"""
    bbox = data['gt_bboxes'].data.numpy()
    label = data['gt_labels'].data.numpy()
    img = data['img'].data.numpy()     # 逆tensor
    img1 = img.transpose(1,2,0)   # 逆chw
    img2 = np.clip((img1 * img_norm_cfg['std'] + img_norm_cfg['mean']).astype(np.int32), 0, 255)  # 逆归一
#        plt.imshow(img2)
    vis_bbox(img2[...,[2,0,1]], bbox, label-1, label_names=classes)  # vis_bbox内部会bgr转rgb，所以这里要用bgr输入

def test_dataloader():
    data_root = '../data/VOCdevkit/'  # 指代ssd目录下的data目录
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
    cfg_train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='VOCDataset',
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            img_scale=(300, 300),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 4)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
            resize_keep_ratio=False))
    
    trainset = get_dataset(cfg_train, VOCDataset)
    dataloader = Dataloader

if __name__ == '__main__':
    test_voc_dataset()
#    test_coco_dataset()
    
        