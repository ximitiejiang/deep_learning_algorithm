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


def save_checkpoint(checkpoint):
    torch.save(checkpoint)


def load_checkpoint(model, filename, map_location=None):
    """加载模型参数：
    inputs
        filename: 可以是torchvision://的形式，则从/.torch/models文件夹加载，如果不存在则从pytorch官网下载
                  比如torchvision://alexnet, torchvision://resnet34，都是合法形式
                  也可以是文件目录加载，比如/home/ubuntu/xxx/xxx.resnet34.pth
        map_location: 用于加载checkpoint时定义加载到cpu还是GPU
    return:
        checkpoint 也就是OrderedDict类型数据
    """
    # 
    if filename.startswith("torchvision://"):
        model_urls = get_torchvision_models()  # 获得所有模型地址: dict形式
        model_name = filename[14:]             # 获得模型地址：也就是去除字符"torchvision://"
        checkpoint = model_zoo.load_url(model_urls[model_name])  # 从model_zoo获得模型预训练参数：下载或者本地加载
    else:
        if not os.path.isfile(filename):
            raise IOError("%s is not a checkpoint file."%filename)
        checkpoint = torch.load(filename, map_location=map_location)  # 从本地路径加载
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
    