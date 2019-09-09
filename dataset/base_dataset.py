#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:32 2019
@author: ubuntu
"""
from torch.utils.data import Dataset


class BasePytorchDataset(Dataset):
    
    def __init__(self):
        pass
    
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

