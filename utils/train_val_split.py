#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:14:43 2019

@author: ubuntu
"""
import csv

def train_val_split(ann_file, train_perc=0.7):
    """用于对训练集进行划分，也可用于生成小样本数据集，输出训练集和验证集, 划分对象是ann_file
    也就是把ann_file分解成train_ann, val_ann
    Args:
        ann_file(str):      csv file path
        train_perc(float)
    Return:
        train_ann_file(csv)
        val_ann_file(csv)
    """
    with open(ann_file) as f:
        lines = csv.read(f)
        for line in lines:
            
    
    
    