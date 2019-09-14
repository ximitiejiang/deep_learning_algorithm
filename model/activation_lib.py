#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 08:42:44 2019

@author: ubuntu
"""
import torch.nn as nn

activation_dict = {'sigmoid':nn.Sigmoid,
                   'relu':nn.ReLU,
                   'elu':nn.ELU,
                   'leaky_relu':nn.LeakyReLU}

class BaseActivation(nn.Module):
    def __init__(self):
        pass
    def __call__(self, x):
        pass