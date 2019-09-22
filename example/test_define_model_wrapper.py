#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:18:51 2019

@author: ubuntu
"""
import torch.nn as nn
import torch


def _scatter(inputs, target_gpus, dim=0):
    """底层scatter函数"""
    pass
    
    
class ModelWrapper(nn.Module):
    """如何手写一个并行计算模型： 参考pytorch的nn.parallel.data_parallel类
    功能：一方面能够进行并行计算，另一方面能够对输入数据送入相应的device。
    
    args:
        model: 原始模型
        device_ids: 表示可以用来做并行计算的设备
        output_device: 表示最终计算完成输出的单个设备，默认应该是第0个设备
    """
    def __init__(self, model, device_ids=None, output_device=None):
        #对传入的模型进行包装
        self.module = model
        # 获得已有设备ids
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        # 获得输出设备
        if output_device is None:
            output_device = device_ids[0]
        
        self.device_ids
        self.output_device
        self.src_device_obj
        
    def forward(self, *inputs, **kwargs):
        # 如果没有gpu设备，则直接由模型输出
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        
        # 把参数也复制多份到多个device
        # 注意：在这个scatter函数中，包含了把输入inputs, kwargs都直接送入设备的功能，
        # 所以也就不需要把数据通过data.to()手动送入device的必要了。
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        # 如果只有一个gpu，则取已经复制多份的参数的其中一份计算即可
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        # 把模型也复制多份到多个device
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        # 计算多个复制模型的输出
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
    
    
    def scatter(self, inputs, kwargs, device_ids, dim=0):
        """用来把参数复制多份到devices，
        注意：这个函数可以涵盖把数据送入相应设备的功能，即把data.to()函数包含在内
        """
        inputs = _scatter(inputs, device_ids, dim) if inputs else []
        kwargs = _scatter(kwargs, device_ids, dim) if kwargs else []
        
    

    
    def replicate(self, inputs, kwargs, device_ids):
        """用来把模型复制多份到devices"""
        pass
    
    def parallel_apply(self, replicas, inputs, kwargs):
        """用来并行计算的过程"""
        pass
    
    def gather(self, outputs, output_device):
        """用来把并行计算的结果汇总成一个结果"""
        pass
        