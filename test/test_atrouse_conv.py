#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:49:41 2019

@author: ubuntu
"""

import torch
import torch.nn as nn
import torchvision

model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
img = torch.randn(1,3,300,300)
out = model(img)

#
#model = nn.Conv2d(3, 3, kernel_size=3, dilation=2, stride=1, padding=0)
#
#img = torch.randn(1, 3, 300, 300)
#out = model(img)   # w = 