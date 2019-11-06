#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:38:08 2019

@author: ubuntu
"""
import os

def get_classes(dataset_name):
    classes_dict = {'voc': VOC_CLASSES,
                    'coco': COCO_CLASSES,
                    'widerface': WIDERFACE_CLASSES,
                    'imagenet': IMAGENET_CLASSES}
    return classes_dict[dataset_name]


# voc数据集
VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

# coco数据集
COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]

# wideface数据集
WIDERFACE_CLASSES = ['face',]

#imagenet数据集
def imagenet_labels(label_file):
    """获得imagenet label"""
    labels = []
    with open(label_file) as f:
        lines = f.readlines()   # 1000行标签
        for line in lines:
            label = line[10:-1]
            labels.append(label)
    return labels

IMAGENET_CLASSES = imagenet_labels(os.path.dirname(os.path.abspath(__file__))+'/imagenet_labels.txt')