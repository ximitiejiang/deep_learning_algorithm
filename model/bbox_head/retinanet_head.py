#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:09:48 2019

@author: ubuntu
"""
import torch.nn as nn


class RetinaNetHead(nn.Module):
    """retina head"""
    def __init__(self, 
                 input_size,
                 num_classes=21,
                 in_channels=256,
                 base_scale=4,
                 loss_cls_cfg=None,
                 loss_reg_cfg=None,
                 **kwargs):
        
        super().__init__()