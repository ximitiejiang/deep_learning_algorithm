#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:54:50 2019

@author: ubuntu
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

class BasePytorchDataset(Dataset):
    
    def __init__(self):
        pass
    
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    

class Cifar10Dataset(BasePytorchDataset):
    """原版数据集地址http://www.cs.toronto.edu/~kriz/cifar.html
    单张图片为RGB 32x32的小图，总计60,000张，其中50,000张训练集，10,000张测试集
    cifar10: 10个类别，每个类别6000张
    cifar100: 100个类别，每个类别600张
    该数据集没有索引，所以只能一次性加载到内存
    输出：n,h,w,c (bgr格式), 所有图片源数据都统一用这种格式(包括voc/coco)
    """
    def __init__(self, root_path='/home/ubuntu/MyDatasets/cifar-10-batches-py/', 
                 data_type='train', img_transform=None, label_transform=None, bbox_transform=None):
        super().__init__()
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.bbox_transform = bbox_transform
        
        train_path = [root_path + 'data_batch_1',
                      root_path + 'data_batch_2',
                      root_path + 'data_batch_3',
                      root_path + 'data_batch_4',
                      root_path + 'data_batch_5']
        test_path = [root_path + 'test_batch']

        if data_type == 'train':
            self.path = train_path
        elif data_type == 'test':
            self.path = test_path
        else:
            raise ValueError('wrong data type, only support train/test.')
        self.meta_path = root_path + 'batches.meta'    
        
        dataset = self.get_dataset()
        self.imgs = dataset['data']
        self.labels = dataset['target']
        self.bboxes = dataset.get('bbox', None)
        self.CLASSES = dataset['target_names']
    
    def get_dataset(self):
        datas = []
        labels = []
        # 获取标签
        with open(self.meta_path, 'rb') as f:  # 参考cifar源网站的python代码
            dict = pickle.load(f, encoding='bytes')
            label_names = dict[b'label_names']
        # 获取数据    
        for path in self.path:
            with open(path, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
                data = dict[b'data']               # (10000, 3072)
                label = np.array(dict[b'labels'])  # (10000,)
            datas.append(data)
            labels.append(label)
        cat_datas = np.concatenate(datas, axis=0)  # (n, 3072)->(50000,3072)
        cat_labels = np.concatenate(labels)        # (n,)->(50000,)
        # 分别提取R/G/B组成(C,H,W): 原始顺序参考官网说明         
        cat_datas = cat_datas.reshape(-1, 3, 32, 32).transpose(0,2,3,1)[...,[2,1,0]]  # (b,c,h,w)->(b,h,w,c), rgb->bgr
        # 按sklearn格式返回数据        
        dataset = {}
        dataset['data'] = cat_datas
        dataset['target'] = cat_labels
        dataset['target_names'] = label_names
        return dataset
    
    def __getitem__(self, idx):
        """常规数据集传出的是多个变量，这里改为传出dict，再在定制collate中处理堆叠
        注意：要求传出的为OrderedDict，这样在自定义collate_fn中不会出错。
        """
        data_dict = OrderedDict
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.bboxes is not None:
            bbox = self.bboxes[idx]
        
        if self.label_transform is not None:
            label = self.label_transform(label)
            data_dict['label'] = label            
        
        if self.img_transform is not None:
            img, ori_shape, scale_factor = self.img_transform(img)  # transform输出img(img, ori_shape, scale_factor), label
            data_dict['img'] = img
            data_dict['ori_shape'] = ori_shape
            data_dict['scale_factor'] = scale_factor
            
        if self.bboxes is not None and self.bbox_transform is not None:
            bbox = self.bbox_transform(bbox)
            data_dict['bbox'] = bbox
            
        return data_dict
    
    def __len__(self):
        return len(self.imgs)


class Cifar100Dataset(Cifar10Dataset):
    """原版数据集地址http://www.cs.toronto.edu/~kriz/cifar.html
    单张图片为RGB 32x32的小图，总计60,000张，其中50,000张训练集，10,000张测试集
    cifar10: 10个类别，每个类别6000张
    cifar100: 100个类别，每个类别600张
    """
    def __init__(self, root_path='../dataset/source/cifar100/', data_type='train',
                 norm=None, label_transform_dict=None, one_hot=None, 
                 binary=None, shuffle=None):
        
        train_path = [root_path + 'train']
        test_path = [root_path + 'test']

        if data_type == 'train':
            self.path = train_path
        elif data_type == 'test':
            self.path = test_path
        else:
            raise ValueError('wrong data type, only support train/test.')
        self.meta_path = [root_path + 'meta']    
        
        dataset = self.get_dataset()
        self.imgs = dataset['data']
        self.labels = dataset['target']
        self.CLASSES = dataset['target_names']
        