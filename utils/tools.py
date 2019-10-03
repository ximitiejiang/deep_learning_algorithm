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
from contextlib import ContextDecorator

# %%
"""创建一个装饰器，用来统计每个函数的运行时间
使用方法：    
@timeit
def func(*args, **kwargs):        
"""

class timer(ContextDecorator):
    """继承上下文管理装饰器实现一个双功能计时器，既可对函数计时，也可对代码块计时。
    继承的ContextDecorator实现了一个内部__call__()已生成了一个装饰器，内部建立上下文语义。
    所以该类是用__call__实现了装饰器计时功能，然后用上下文管理器实现代码块计时。
    
    time库的两个函数: time.time()返回的1970开始计时后的float秒数.
    time.localtime(time.time())返回struct(year,month,day,hour,min,sec,xx,xx,xx)共9个int数.
    使用方式1:
        @timer('id1')
        def fun()
    使用方式2:
        with timer('id2'):
    """
    def __init__(self, time_id):
        self.time_id = time_id
    
    def __enter__(self):        # 上下文管理器：with执行前执行
        self.start = time.time()
        
    def __exit__(self, *args):  # 上下文管理器：with执行后执行。
        elapse = time.time() - self.start
        print('timer: %s elapse %.3f seconds.'%(self.time_id, elapse))


@timer('aa')
def hello(sec):
    for i in range(5):
        time.sleep(sec)

def hello2(sec):
    with timer('position1'):
        for i in range(5):
            time.sleep(sec)
    
    
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
    hello(1)
    hello2(2)