#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 07:36:02 2019

@author: ubuntu
"""
import os
import cv2
import numpy as np
from PIL import Image
from dataset.voc_dataset import VOCDataset


class CityScapesDataset(VOCDataset):
    """主要是分割数据集，特点是以道路交通为主，且包含一部分非常精细标注的数据
    所下载的数据集不是官方数据集，而是经过处理的，来自https://blog.csdn.net/zym19941119/article/details/81198315
    1. 可以用核心的精细标注数据做分割任务(gtFine)：这是当前数据集处理程序选择的方式，包含精细图片训练集2975张，验证集500张
        - img_ann采用trainImages.txt
        - seg_ann通过img_path变换而来，所以不需要提供
        
    2. 也可以考虑用普通标注的数据做分割任务(gtFine + leftImg8bit): 
        - img_ann采用trainImages.txt + trainInstances.txt
    """
    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def load_annotations(self, ann_file):
        """可加载多个标注文件
        """
        img_anns = []
        for i, af in enumerate(ann_file): # 分别读取多个子数据源
            with open(af) as f:
                img_ids = f.readlines()
            for j in range(len(img_ids)):
                img_ids[j] = img_ids[j][:-1]  # 去除最后的\n字符
            # 基于图片id打开annotation文件，获取img/xml文件名
            for img_id in img_ids:
                img_file = self.img_prefix[i] + img_id
                seg_file = self.img_prefix[i] + 'gtFine' + img_id[11:]
                seg_file = seg_file.replace('leftImg8bit', 'gtFine_labelIds')
                # 配对img和seg文件
                if os.path.isfile(img_file) and os.path.isfile(seg_file):
                    img_anns.append(dict(img_file = img_file, 
                                         seg_file = seg_file))
                else:
                    continue
        return img_anns
    
    def __getitem__(self, idx):
        """虽然ann中有解析出difficult的情况，但这里简化统一没有处理difficult的情况，只对标准数据进行输出。
        """
        data = {}
        ori_shape = None
        scale_shape = None
        pad_shape = None
        scale_factor = None
        flip = None
        # 读取图片
        img_info = self.img_anns[idx]
        img_path = img_info['img_file']
        img = cv2.imread(img_path)
        
        # basic transform
        if self.img_transform is not None:    
            # img transform
            img, ori_shape, scale_shape, pad_shape, scale_factor, flip = self.img_transform(img)
        
        # 组合img_meta
        img_meta = dict(ori_shape = ori_shape,
                        scale_shape = scale_shape,
                        pad_shape = pad_shape,
                        scale_factor = scale_factor,
                        flip = flip)
        # 组合数据: 注意img_meta数据无法堆叠，尺寸不一的img也不能堆叠，所以需要在collate_fn中自定义处理方式
        data.update(img = img,
                    img_meta = img_meta,
                    stack_list = ['img'])
        
        # 如果是分割任务，提供的是分割png，所以用的是seg_transform
        if self.seg_transform is not None and img_info.get('seg_file') is not None:
            seg_path = self.img_anns[idx]['seg_file']
            seg = Image.open(seg_path)   # 采用PIL.Image读入图片可以直接得到用0-33表示像素同事表示类别的灰度图
            # 确保seg作为标签必须为int64(long)
            seg = np.asarray(seg) # (h,w)
            seg = self.seg_transform(seg, flip).long()  # 类似于对img的变换，只需输入seg，额外一个从img transform来的参数，保证与img一致
            data.update(seg = seg)
            data['stack_list'].append('seg')

        return data

    def __len__(self):
        return len(self.img_anns)