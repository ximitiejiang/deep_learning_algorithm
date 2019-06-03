#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:08:06 2019

@author: ubuntu
"""
from sklearn import datasets

class SklearnDataset():
    def __init__(self):
        pass
    
    @staticmethod
    def load(dset_name):
        """"dset_name = ['digits','boston', 'iris','diabets','linnerud']
        分别代表[0-9手写数字集，波士顿房价集，鸢尾花数据集，糖尿病人数据集，健身男子指标数据集]
        """
        dset_name = dset_name
        
if __name__ == "__main__":
    dataset = SklearnDataset.load('digits')
    