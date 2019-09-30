#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:41:38 2019

@author: ubuntu
"""
from utils.prepare_training import get_config, get_dataset

def calc_dataset_bbox_area(cfg_path):
        
    cfg = get_config(cfg_path)
    trainset = get_dataset(cfg.trainset, cfg.transform)
    
    areas = []
    labels = []
    for data in trainset:
        img_meta = data['img_meta']
        gt_labels = data['gt_labels']
        gt_bboxes = data['gt_bboxes']
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        area = w * h
        areas.extend(area)
        labels.extend(gt_labels)
    
if __name__ == '__main__':
    cfg_path = ''
    calc_dataset_bbox_area(cfg_path)