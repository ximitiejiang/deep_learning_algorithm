#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:36:59 2019

@author: ubuntu
"""
from model.backbone.vgg_lib import VGG
from utils.init_weights import common_init_weights

class FCNVGG16(VGG):
    """分割模型FCN8s, FCN16s, FCN32s的backbone"""
    def __init__(self, 
                 depth=16, 
                 pretrained=None, 
                 out_indices=(16, 23, 30)):
        self.depth = depth
        self.pretrained = pretrained
        self.out_indices = out_indices
        super().__init__(depth=depth, pretrained=pretrained)
        
    
    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    
    def init_weights(self):
        common_init_weights(self, pretrained=self.pretrained)