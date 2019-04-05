#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:19:47 2019

@author: suliang
"""
import logging

import torch
import torch.nn as nn
from .vgg import VGG
from .weight_init import constant_init, normal_init, kaiming_init
from .checkpoint import load_checkpoint
from utils.registry_build import registered


@registered.register_module
class M2detVGG(VGG):
    """基于VGG的简版，只用于feature extract, 提取22/34的两层
    """
    def __init__(self,
                 input_size,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,        # 针对maxpool层的取整参数
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 l2_norm_scale=20.,
                 **kwargs):   # 添加一个**kwargs: 有一个type没地方放，又不想改cfg
        super(M2detVGG, self).__init__(
            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices)
        assert input_size in (300, 512)
        self.input_size = input_size

        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.out_feature_indices = out_feature_indices

        self.inplanes = 1024

        self.l2_norm = L2Norm(
            self.features[out_feature_indices[0] - 1].out_channels,
            l2_norm_scale)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        constant_init(self.l2_norm, self.l2_norm.scale)  # l2 norm参数没有预训练值，需要单独初始化

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)
                
        outs[0] = self.l2_norm(outs[0])
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return self.weight[None, :, None, None].expand_as(x) * x / norm


if __name__=="__main__":
    m2detvgg = M2detVGG(input_size=300,
                    depth=16,
                    with_last_pool=False,
                    ceil_mode=True,
                    out_indices=(3, 4),
                    out_feature_indices=(22, 34),
                    l2_norm_scale=20.)
    print(m2detvgg)
    m2detvgg.init_weights(pretrained = 'weights/m2det/vgg16_reducedfc.pth')  # 新版mmcv中load checkpoint支持下载weights了
                                                                  # 可以看到加载的state dict只包含vgg本体的，新增的extra layer/l2 norm不包含
    
    