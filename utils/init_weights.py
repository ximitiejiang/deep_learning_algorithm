import torch.nn as nn
import numpy as np
from utils.checkpoint import load_checkpoint

def common_init_weights(model, pretrained=None, map_location=None):
    """通用的模型初始化函数"""
    if isinstance(pretrained, str):
        load_checkpoint(model, pretrained, map_location = map_location)
    
    elif pretrained is None:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
    else:
        raise TypeError('pretrained must be a str or None')

def constant_init(module, val, bias=0):
    """用常数初始化："""
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """用xavier"""
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """用正态分布初始化，默认是标准正态分布N(0,1)"""
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    """用平均分布[0,1]方式初始化"""
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, mode='fan_out', nonlinearity='relu',
                 bias=0, distribution='normal'):
    """用kaiming初始化，其分布可以选择正态分布或者平均分布"""
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def bias_init_with_prob(prior_prob):
    """ 初始化偏置值基于给定的概率: 用在fcos算法中给卷积的偏置做初始化"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init