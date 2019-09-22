#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:27:21 2019

@author: ubuntu
"""

import torch
from torch.autograd import Function

"""如何在pytorch中手写一个Function类，来利用pytorch的自动求导功能。
从而可以用来实现前向计算和反向传播，也就可以用来自定义层。
参考自：叠加态的猫 https://www.cnblogs.com/hellcat/p/8453615.html
"""
class Sigmoid(Function):
                                                              
    @staticmethod
    def forward(ctx, x):  # ctx类似于Function类的self, 用来在forward函数好backward函数之间共享参数，只需要把要共享的参数通过命令ctx.save_for_backward保存，就可以在backward中通过ctx.saved_tensors()获得。
        output = 1 / (1 + torch.exp(-x))  # 
        ctx.save_for_backward(output)
        return output
         
    @staticmethod
    def backward(ctx, grad_output): # 这里grad_output就相当于accum_grad，是前一级的输出梯度
        output,  = ctx.saved_tensors  # 新版pytorch用saved_tensors替代了saved_variables
        grad_x = output * (1 - output) * grad_output
        return grad_x                           
 
# 采用数值逼近方式检验计算梯度的公式对不对
#test_input = V(torch.randn(3,4), requires_grad=True)  # 早期版本需要用Varaiable
test_input = torch.randn(3, 4, requires_grad=True)
torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-3)


def f_sigmoid(x):
    y = Sigmoid.apply(x)
    y.backward(torch.ones(x.size()))
     
def f_naive(x):
    y =  1/(1 + torch.exp(-x))
    y.backward(torch.ones(x.size()))
     
def f_th(x):
    y = torch.sigmoid(x)
    y.backward(torch.ones(x.size()))
     
x=torch.randn(100, 100, requires_grad=True)
#%timeit -n 100 f_sigmoid(x)
#%timeit -n 100 f_naive(x)
#%timeit -n 100 f_th(x)