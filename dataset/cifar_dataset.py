#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:54:50 2019

@author: ubuntu
"""
import pickle
import numpy as np
from dataset.base_dataset import BaseDataset
##########################################
# Dataset Interface:                     #
#   self.datas, self.labels              #
#   self.feat_names, self.label_names    #
#   self.classes                         #
#   self.num_classes, self.num_features  #
##########################################

class Cifar10Dataset(BaseDataset):
    """原版数据集地址http://www.cs.toronto.edu/~kriz/cifar.html
    单张图片为RGB 32x32的小图，总计60,000张，其中50,000张训练集，10,000张测试集
    cifar10: 10个类别，每个类别6000张
    cifar100: 100个类别，每个类别600张
    """
    def __init__(self, root_path='../dataset/source/cifar10/', data_type='train',
                 norm=None, label_transform_dict=None, one_hot=None, 
                 binary=None, shuffle=None):
        
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
        
        super().__init__(norm=norm, 
                         label_transform_dict=label_transform_dict, 
                         one_hot=one_hot,
                         binary=binary,
                         shuffle=shuffle) # 先准备self.path再init中调用get_dataset()
    
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
        # 分别提取R/G/B组成(C,H,W):         
        cat_datas = cat_datas.reshape(-1, 3, 32, 32)  # (b, c, h, w) 其中c为rgb
        # 按sklearn格式返回数据        
        dataset = {}
        dataset['data'] = cat_datas
        dataset['target'] = cat_labels
        dataset['target_names'] = label_names
        return dataset


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
        
        super().__init__(norm=norm, 
                         label_transform_dict=label_transform_dict, 
                         one_hot=one_hot,
                         binary=binary,
                         shuffle=shuffle) # 先准备self.path再init中调用get_dataset()
        