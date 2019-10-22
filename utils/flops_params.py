#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:35:09 2019

@author: ubuntu
"""
import torch
from thop import profile
from thop import clever_format
from torchvision.models import resnet50

"""
采用pytorch的第三方库thop计算模型的参数量和计算量
thop来自: https://github.com/Lyken17/pytorch-OpCounter
thop安装: pip install thop
"""

img = torch.randn(1, 3, 640, 640)
model = resnet50()
flops, params = profile(model, inputs=(img,))
flops, params = clever_format([flops, params], '%.3f')

print('params:', params, 'flops', flops)