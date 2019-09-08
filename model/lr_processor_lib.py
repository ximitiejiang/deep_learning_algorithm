#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:28:30 2019

@author: ubuntu
"""
import copy

class LrProcessor():
    """学习率调整器"""
    def __init__(self, runner, warmup_type=None, 
                 warmup_iters=0, warmup_ratio=0.1,
                 **kwargs):
        self.runner = runner
        self.warmup_type = warmup_type
        self.warmup_iters = warmup_iters  # 热身次数
        self.warmup_ratio = warmup_ratio
        # 均以group list的形式表示lr
        self.base_lr_group = []     # 基础学习率组(不变)：可能等于optimizer预定义的不可变学习率，也可能等于resume后的学习率
        self.regular_lr_group = []  # 常规学习率组(动态): 每种LrProcessor都会动态调整的部分
    
    def get_warmup_lr_group(self, current_iter):    
        if self.warmup_type == 'constant': # 相当于用一个固定的小学习率热身
            warmup_lr_group = [lr * self.warmup_ratio for lr in self.regular_lr_group]
        if self.warmup_type == 'linear':   # 
            k = (1 - current_iter / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr_group = [lr * (1 - k) for lr in self.regular_lr_group]
        if self.warmup_type == 'exp':
            k = self.warmup_ratio**(1 - current_iter / self.warmup_iters)
            warmup_lr_group = [_lr * k for _lr in self.regular_lr_group]
        return warmup_lr_group
            
    def get_regular_lr_group(self):
        """生成常规学习率组"""
        regular_lr_group = []
        for base_lr in self.base_lr_group:
            regular_lr_group.append(self.get_regular_lr(self.runner, base_lr))   # 子类需要实现对单个学习率计算的函数，到这里汇总成regular_lr group的list形式
        return regular_lr_group
        
    def get_regular_lr(self, runner, base_lr):
        raise NotImplementedError
    
    def _set_lr(self, lr_groups):
        """唯一的调整学习率的底层函数，其他函数都调用这个函数进行学习率修改
        lr_groups: 用于填充到optimizer去的自定义学习率组
        """
        for group, lr in zip(self.runner.optimizer.param_groups, lr_groups):
            group['lr'] = lr
        
    def set_base_lr_group(self):
        """第一步：在训练开始或者恢复开始阶段，初始化base_lr_group
        不能放在模型初始化时做，因为恢复训练模式下不会进行模型初始化，所以要确保在开始前向计算之前操作
        """
        self.base_lr_group = [group['lr'] for group in self.runner.optimizer.param_groups]  # 以优化器的参数为准，所以保存模型并恢复模型时，需要连优化器一起保存    
    
    def set_regular_lr_group(self):
        """第二步：在每个epoch开始前，先获得regular_lr_group，并填入optimizer
        """
        self.regular_lr_group = self.get_regular_lr_group()
        self._set_lr(self.regular_lr_group)
    
    def set_warmup_lr_group(self):
        """第三步：在第一个epoch前的部分iters设置warmup_lr_group
        """
        current_iter = self.runner.c_iter   # 代表从0开始
        current_epoch = self.runner.c_epoch # 代表从0开始
        if current_epoch == 0:
            warmup_lr_group = self.get_warmup_lr_group(current_iter)
            if current_iter < self.warmup_iters:      # 如果没有超过热身次数，则设置为warmup_lr
                self._set_lr(warmup_lr_group)
            elif current_iter == self.warmup_iters:   # 如果初次超过热身次数，则设置学习率为常规学习率
                self._set_lr(self.regular_lr_group)
            elif self.warmup_type is None or current_iter > self.warmup_iters:
                return

class FixedLrProcessor(LrProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_regular_lr(self, runner, base_lr):
        return base_lr


class ListLrProcessor(LrProcessor):
    """学习率按列表阶梯下降，每个阶梯下降的lr查表得到(此时跟base_lr无关了)
    例如：step=[2,4], lr=[0.005, 0.0001], 则在第2个epoch调整lr到0.005，第4个epoch调整到0.0001
    """
    def __init__(self, step=None, lr=None, **kwargs):
        super().__init__(**kwargs)
        self.step = step
        self.lr = lr
        self.lr_internal = copy.copy(lr)  # 深拷贝
        
    def get_regular_lr(self, runner, base_lr):
        if len(self.lr_internal) == len(self.lr):
            self.lr_internal.insert(0, base_lr)
        current_epoch = runner.c_epoch
        for i, ep in enumerate(self.step):
            if current_epoch + 1 < ep:  # 加1表示epoch个数
                pos = i  # 小于只要找到最小的，也就是第一个
                break
            if current_epoch + 1 >= ep:
                pos = i + 1  # 大于需要找到最大的，也就是最后一个
        return self.lr_internal[pos]


class StepLrProcessor(LrProcessor):
    """学习率阶梯式下降，每一阶段固定下降1/10
    例如：step = [2, 4]，则在第2个epoch调整lr到0.1*lr, 第4个epoch调整学习率到0.01*lr
    """
    def __init__(self, step=None, gamma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.step = step
        self.gamma = gamma
        
    def get_regular_lr(self, runner, base_lr):
        current_epoch = runner.c_epoch  # 代表从0开始  
        exp = len(self.step)  # 初始为n
        for i, ep in enumerate(self.step):
            if current_epoch + 1 < ep:  # 这里加1表示epoch个数
                exp = i
                break
        return base_lr * self.gamma**exp