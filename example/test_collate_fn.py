#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:47:53 2019

@author: ubuntu
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


from torch.utils.data.dataloader import default_collate
def multi_collate(batch):
    """自定义一个多数据分别堆叠的collate函数：
    参考：https://www.jianshu.com/p/bb90bff9f6e5
    原有的collate_fn可以对输入的多种数据进行堆叠，但堆叠行为是stack模式，
    输入[dataset[i] for i in batch_indices], 也就是所有batch data打包的list.
    也就是img(3,32,32)->(64,3,32,32), label(1)->[64], shape(3)
    堆叠方式也比较简单粗暴，就是增加一个维度比如64个(3,32,32)变成(64,3,32,32),
    如果需要传入其他数据比如bbox, scale，则需要自定义collate_fn
    
    输入：batch为list，是DataLoader提供了通过getitem获得的每个batch数据，list长度就是batch size.
    输出：batch也为list，但每个变量都是堆叠好的。
    """
    result = []
    for sample in zip(*batch):  # 参考pytorch源码写法：数据先分组提取
        if isinstance(sample[0], torch.Tensor):
            stacked = torch.stack(sample, dim=0)    
            result.append(stacked)
        if isinstance(sample[0], np.ndarray):
            stacked = np.stack(sample, axis=0)
            result.append(torch.tensor(stacked))
        if isinstance(sample[0], (int, float)):
            stacked = np.stack(sample, axis=0)
            result.append(torch.tensor(stacked))
    return result

def test1():
    """测试4种不同类型数据的处理方式
    1. 标量，升级为1维数组(numpy, tensor都一样)
    2. 一维数组，升级为2维数组
    """
    img = torch.ones(2,4)
    label = torch.tensor(2)
    shape = np.array([3,16,16])
    scale = 1
    # 这里模拟了一个batch的数据，包含3个样本
    batch = [[img, label, shape, scale], [img, label, shape, scale], [img, label, shape, scale]]
    result = []
    for sample in zip(*batch):  
        if isinstance(sample[0], torch.Tensor):
            stacked = torch.stack(sample, dim=0)    
            result.append(stacked)
        if isinstance(sample[0], np.ndarray):
            stacked = np.stack(sample, axis=0)
            result.append(torch.tensor(stacked))
        if isinstance(sample[0], (int, float)):
            stacked = np.stack(sample, axis=0)
            result.append(torch.tensor(stacked))
    return result


# %% 
def dict_collate(batch):
    """batch的获取是通过[dataset[i] for i in indics]，所以必然是list
    但里边dataset返回的每一个数据，可以是dict, 比如{'img': img, 'label': label, 'shape': shape}
    """
    result = []
    return result

def test3():
    
    img1 = torch.ones(3, 10, 10)
    img2 = torch.ones(3, 15, 20)
    img3 = torch.ones(3, 12, 30)
    
    label = torch.tensor([1,2,9])
    shape = np.array([3,16,16])
    scale = 1
    meta = dict(ori_shape=(32,32,3),
                pad_shape=(43,43,3))
    # 假定是OrderedDict
    batch = [{'img':img1, 'label':label, 'shape':shape, 'scale':scale, 'meta':meta, 'stack_list':['img']},
             {'img':img2, 'label':label, 'shape':shape, 'scale':scale, 'meta':meta, 'stack_list':['img']},
             {'img':img3, 'label':label, 'shape':shape, 'scale':scale, 'meta':meta, 'stack_list':['img']}]
    
    result = dict()
    data = batch[0].values()
    data = list(data)
    stack_list = batch[0]['stack_list']
    for i, name in enumerate(batch[0].keys()):  # 第i个变量的堆叠
        if name in stack_list:
            data_list = [sample[name] for sample in batch]
            shape_stack = np.stack([data.shape for data in data_list], axis=0)
            max_c, max_h, max_w =  np.max(shape_stack, axis=0)
            stacked = torch.zeros(len(batch), max_c, max_h, max_w)  # b,c,h,w
            for dim in range(len(batch)):
                da = data_list[dim]
                stacked[dim,:da.shape[0],:da.shape[1],:da.shape[2]] = da
            result[name] = stacked
            
        else:  
            result[name] = [sample[name] for sample in batch]
    return result  # 期望的result应该是{'img': img, 'label':label}

# %%
def to_device(data, device):
    """用于把数据送入device: 可一次性把一个list中的tensor都送入同一device"""
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    if isinstance(data, list) and isinstance(data[0], torch.Tensor):
        data = [data[i].to(device) for i in range(len(data))]
    return data

# %%
if __name__ == "__main__":
    
    name = 'to_device'
    
    if name == 'test':
        result = test3()
        result['img'].shape
        len(result['meta'])
    
    if name == 'to_device':
        device = torch.device('cpu')
        a = torch.tensor([1,2])
        b = torch.tensor([3,4])
        c = [a, b]
    
        d = to_device(a, device)
        e = to_device(c, device)
    
    