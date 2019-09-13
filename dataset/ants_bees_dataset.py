#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:17:04 2019

@author: ubuntu
"""
import os
import numpy as np
import cv2
from dataset.base_dataset import BasePytorchDataset

class AntsBeesDataset(BasePytorchDataset):
    """简易二分类数据集：蚂蚁和蜜蜂
    数据集分两个文件夹：
        - train: 里边包含ants, bees两个文件夹, 共计123+121=244张图片
        - val: 里边包含ants, bees两个文件夹
    """
    def __init__(self, root_path, 
                 img_transform=None,
                 label_transform=None,
                 bbox_transform=None,
                 aug_transform=None,
                 data_type=None):
        self.data_type = data_type
        # 变换函数
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.bbox_transform = bbox_transform
        self.aug_transform = aug_transform
        if data_type == 'train':
            self.img_path = root_path + 'train/'
        elif data_type == 'val':
            self.img_path = root_path + 'val/'
        
        self.img_anns = self.get_ann_info(self.img_path)  # (244,)
       
    def get_ann_info(self, img_path):
        class_labels = {'ants':0, 'bees':1}
        ann_list = []
        for class_name, label in zip(class_labels.keys(), class_labels.values()):
            prefix = img_path + class_name
            img_list = os.listdir(prefix)
            img_list = [name for name in img_list if not name.startswith('.') and not name.endswith('gif')]

            ann = {}
            for img_name in img_list:
                ann['img_path'] = prefix + '/' + img_name
                ann['label'] = label
                ann_list.append(ann)
        # 打乱顺序
        inds = np.arange(len(ann_list))
        shuffered = np.random.permutation(inds)
        ann_array = np.array(ann_list)[shuffered]
        return ann_array
        
        
    def __getitem__(self, idx):
        """分类数据集，输出为data dict: {'img':img, 'label':label}
        """
        ann = self.img_anns[idx]
        img = cv2.imread(ann['img_path']) # hwc
        label = ann['label']
        
        data_dict = {}
        
        if self.label_transform is not None:
            label = self.label_transform(label)
        data_dict['label'] = label            
        
        if self.img_transform is not None:
            img, ori_shape, scale_factor = self.img_transform(img)  # transform输出img(img, ori_shape, scale_factor), label
            data_dict['ori_shape'] = ori_shape
            data_dict['scale_factor'] = scale_factor
        data_dict['img'] = img

        return data_dict
     
    def __len__(self):
        return len(self.img_anns)


if __name__ == "__main__":
    ab = AntsBeesDataset(root_path = '/home/ubuntu/MyDatasets/AntsBees/', data_type='val')
    data = ab[0]
    img = data['img']
    label = data['label']