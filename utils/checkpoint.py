#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:42:56 2019

@author: ubuntu
"""
from importlib import import_module
import pkgutil
import torch
import torchvision
from torch.utils import model_zoo
import os
from collections import OrderedDict

from utils.tools import exist_or_mkdir

def save_checkpoint(path_with_name, model, optimizer, meta):
    """保存模型：无论之前是什么模型，一律保存成cpu模型，便与统一处理
    """
    # 为确保能保存不报错，无论目录是否存在都创建目录并保存。
    dirname = os.path.dirname(path_with_name)
    exist_or_mkdir(dirname)
    # 如果是DataParalle model，则去掉外壳module
    if hasattr(model, 'module'):
        model = model.module
    # 保存元数据和模型状态字典
    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())}
    # 保存优化器状态字典
    checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, path_with_name)  
    
    
def weights_to_cpu(state_dict):
    """把模型参数转换成cpu版本
    """
    state_dict_cpu = OrderedDict()
    for key, value in state_dict.items():
        state_dict_cpu[key] = value.cpu()   # value.cpu()即把权值转换为cpu版本
    return state_dict_cpu


def load_checkpoint(model, checkpoint_path, map_location=None):
    """加载模型参数：
    inputs
        filename: 可以是torchvision://的形式，则从/.torch/models文件夹加载，如果不存在则从pytorch官网下载
                  比如torchvision://alexnet, torchvision://resnet34，都是合法形式
                  也可以是文件目录加载，比如/home/ubuntu/xxx/xxx.resnet34.pth
        map_location: 用于加载checkpoint时定义加载到cpu还是GPU
    return:
        checkpoint 也就是OrderedDict类型数据
    """
    # 从在线获取pytorch的模型参数：如果已经下载则从本地.torch文件夹直接加载
    if checkpoint_path.startswith("torchvision://"):
        model_urls = get_torchvision_models()  # 获得所有模型地址: dict形式
        model_name = checkpoint_path[14:]             # 获得模型地址：也就是去除字符"torchvision://"
        checkpoint = model_zoo.load_url(model_urls[model_name])  # 从model_zoo获得模型预训练参数：下载或者本地加载
    # 从本地获取模型参数
    else: 
        if not os.path.isfile(checkpoint_path):
            raise IOError("%s is not a checkpoint file."%checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)  # 从本地路径加载
    # 获取state_dict
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module'):  # 如果是并行模型则去掉参数前面的module字段
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # 加载参数到模型
    if hasattr(model, 'module'): # 如果是并行模型
        model.module.load_state_dict(state_dict, strict=False)
    else:   # 如果是常规模型
        model.load_state_dict(state_dict, strict=False)
    return checkpoint


def get_torchvision_models():
    """这是pytorch的标准函数，用于获取所有pytorch的模型下载地址"""
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module('torchvision.models.{}'.format(name))
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


if __name__ == "__main__":
    import torch.nn as nn
    class TestModel(nn.Module):
        def __init__(self):
            pass
    
    model = TestModel()
    load_checkpoint(model, "torchvision://resnet50")
    