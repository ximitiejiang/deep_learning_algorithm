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
from terminaltables import AsciiTable

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
        checkpoint_path: 可以是torchvision://的形式，则从/.torch/models文件夹加载，如果不存在则从pytorch官网下载
                  比如torchvision://alexnet, torchvision://resnet34，都是合法形式
                  也可以是文件目录加载，比如/home/ubuntu/xxx/xxx.resnet34.pth
        map_location: 用于加载checkpoint时定义加载到cpu还是GPU, 通常传入自定义的device
    return:
        checkpoint 也就是OrderedDict类型数据
    """
    if map_location is None:
        map_location = torch.device('cpu')
    # 从在线获取pytorch的模型参数：如果已经下载则从本地.torch文件夹直接加载
    if checkpoint_path.startswith("torchvision://"):
        model_urls = get_torchvision_models()  # 获得所有模型地址: dict形式
        model_name = checkpoint_path[14:]             # 获得模型地址：也就是去除字符"torchvision://"
        checkpoint = model_zoo.load_url(model_urls[model_name])  # 从model_zoo获得模型预训练参数：下载或者本地加载
    # 从本地获取模型参数
    else: 
        if not os.path.isfile(checkpoint_path):
            raise IOError("%s is not a checkpoint file."%checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    # 获取state_dict
    if isinstance(checkpoint, OrderedDict):  # 如果是pytorch原有模型，即ordereddict格式，则直接为state dict
        state_dict = checkpoint
    elif isinstance(checkpoint, dict):        # 如果是自定义的checkpoint dict, 则从中取state dict
        state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module'):  # 如果是并行模型则去掉参数前面的module字段
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # 加载参数到模型
    if hasattr(model, 'module'): # 如果是并行模型
#        model.module.load_state_dict(state_dict, strict=False)
        load_state_dict(model.module, state_dict)
    else:   # 如果是常规模型
#        model.load_state_dict(state_dict, strict=False)
        load_state_dict(model, state_dict)
    return checkpoint


def load_state_dict(model, state_dict, logger=None):
    """用于给model加载state_dict: 剔除命名不同的key和size不同的key，其他则copy给模型参数
    Args:
        model: 需要加载state_dict的梦想欧诺个
        state_dict: 已准备好的state_dict
    """
    my_state_dict = model.state_dict()
    
    not_need_keys = []
    size_mismatch_keys = []
    err_msg = []
    for key, param in state_dict.items():
        if key not in my_state_dict.keys():   #如果key不匹配，则舍弃，跳出该轮循环
            not_need_keys.append(key)
            continue
        if param.size() != my_state_dict[key].size():  # 如果尺寸不匹配，则舍弃，跳出该轮循环
            size_mismatch_keys.append([key, my_state_dict[key].size(), param.size()])
            continue
        my_state_dict[key].copy_(param)   # 其他情况，则复制参数给模型
        
    # 通知用户加载差异：
    all_missing_keys = set(my_state_dict.keys()) - set(state_dict.keys())
    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]
    err_msg = []
    table = AsciiTable([''])  # 初始为空字符
    if not_need_keys:
        err_msg.append('unexpected key in source state_dict: {}'.format(
            ', '.join(not_need_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}'.format(
            ', '.join(missing_keys)))
    if size_mismatch_keys:
        mismatch_info = 'these keys have mismatched shape: '
        
        header = ['key', 'expected shape', 'loaded shape']
        table_data = [header] + size_mismatch_keys
        table = AsciiTable(table_data)
        err_msg.append(mismatch_info)
    else:
        header = ['all the keys sizes matched exactly.']
        table = AsciiTable([header])
    if logger is not None:
        logger.warning(err_msg)
        logger.warning(table.table)
    else:
        print(err_msg)
        print(table.table)
    

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
    