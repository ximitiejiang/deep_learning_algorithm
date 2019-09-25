#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:24:47 2019

@author: ubuntu

该process checker用于模型在运行过程中的各种参数分析和评价

采用类装饰器模型作为基类
class Deco():
    def __init__(self, func):
        print('func name is {}'.format(func.__name__))
        self._func = func
    def __call__(self, *args, **kwargs):
        print('this is class decorator with new function')
        return self._func(*args, **kwargs)
"""

from abc import ABCMeta, abstractmethod
    
# %% 方案1：采用装饰器类作为基类
class CheckDeco():
    __metaclass__ = ABCMeta
    
    def __init__(self, func):
        self._func = func
    
    def __call__(self, me_instance, *args, **kwargs):
        self.forward()
        return self._func(me_instance, *args, kwargs)
    
    @abstractmethod
    def forward(self):
        pass


class CheckBbox(CheckDeco):
    """用于提取和评估模型在整个设计结构上，各个关键部位数据的性能
    """
    def forward(self):
        print("checkbbox is running!")
        

# %% 方案2：采用装饰器函数作为基本方法
def check_bbox(func):
    """一个单独的装饰器
    注意，对于类的方法的装饰器，形参self需要用me_instance代替。对于有返回值的类方法，装饰器也需要返回值。
    """
    def wrapper(me_instance, *args, **kwargs):
        print("check bbox is running...")
        results = func(me_instance, *args, **kwargs)
        return results
    return wrapper


