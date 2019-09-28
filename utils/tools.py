#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:43:44 2019

@author: ubuntu
"""
import torch
import os
import time
import six
import pickle

# %%
"""创建一个装饰器，用来统计每个函数的运行时间
使用方法：    
@timeit
def func(*args, **kwargs):        
"""
def timeit(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
    

# %%

def accuracy(y_pred, label, topk=1):
    """pytorch tensor版本的精度计算：由于都是未概率化的数据，
    y_pred(b, n_classes)，tensor为未概率化的数据
    label(b,), tensor为未概率化的数据(也就是实际标签而不是独热编码)
    输出: acc (float标量)
    """
    with torch.no_grad():
        # TODO: 增加topk的功能
        if topk == 1:
            pred = torch.argmax(y_pred, dim=1)         # 输出(b,)           
            acc = (pred == label).sum().float() / len(label)
        return acc
    

def get_time_str():
    """计算系统时间并生成字符串"""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def exist_or_mkdir(dir_name, mode=0o777):
    """检查目录是否存在，如果不存在则创建: 可创建嵌套文件夹
    """
    dir_name = os.path.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)


# %%            
def save2pkl(var, path, to_numpy=True):
    """保存变量为.pkl文件: 输入的var变量可以是tensor，也可以是list(tensor)
    """
    if not path.endswith('.pkl') or not os.path.isdir(os.path.dirname(path)):
        raise ValueError('path format is not pkl or parent dir is not exist.')
    # 如果要转换成numpy格式
    if to_numpy:
        if isinstance(var, list) and isinstance(var[0], torch.Tensor):
            for i in range(len(var)):
                var[i] = var[i].cpu().numpy() # 如果是tensor，则先转cpu和numpy
        elif isinstance(var, torch.Tensor):
            var = var.cpu().numpy()
    # 保存
    with open(path, 'wb') as f:
        pickle.dump(var, f)


def loadvar(path):
    """采用pickle模块从pkl文件读取变量清单，但需要指定有几个变量在文件中"""
    with open(path, 'rb') as f:
        var_list = pickle.load(f)
    return var_list


# %%
from utils.visualization import vis_loss_acc
def parse_log(path, show=True):
    """解析log文件"""
    with open(path) as f:
        lines = f.readlines()
        data_dict = {'loss': [],
                     'acc': []}
        lines = lines[2:]
        for line in lines:
            loss = float(line.split('\t')[-3].split(' ')[-1])
            acc = float(line.split('\t')[-2].split(' ')[-1])
            data_dict['loss'].append(loss)
            data_dict['acc'].append(acc)
    
    if show:
        vis_loss_acc(data_dict)
    return data_dict


# %%

if __name__ == "__main__":
    path = '/home/ubuntu/mytrain/test.pkl'
    num = [torch.tensor([1,2,3]), torch.tensor([4,5,6])]
    save2pkl(num, path, False)
    aa = loadvar(path)