#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:47:36 2019

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


def vis_loss_acc(buffle, title='train result'):
    """可视化loss和acc
    输入:buffle为dict格式，dict(loss=[], acc=[])
    """
    losses = buffle['loss']
    accs = buffle['acc']
    
    if title is None:
        prefix = ""
    else:
        prefix = title
    assert losses is not None and len(losses) != 0, 'can not visualize losses because losses is empty.'
    
    if isinstance(losses[0], list) or isinstance(losses[0], tuple):  # 如果losses列表里边包含idx
        x = np.array(losses)[:,0]
        y_loss = np.array(losses)[:,1]
    else:  # 如果losses列表里边不包含idx只是单纯loss数值
        x = np.arange(len(losses))
        y_loss = np.array(losses)
    # 绘制loss
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title(prefix + ' losses & acc')
    ax1.set_ylabel('loss')
    lines = ax1.plot(x,y_loss, 'r', label='loss')
    # 绘制acc
    if isinstance(accs[0], list) or isinstance(accs[0], tuple):
        x = np.array(accs)[:,0]
        y_acc = np.array(accs)[:,1]
    else:
        x = np.arange(len(accs))
        y_acc = np.array(accs)
        
    ax2 = ax1.twinx()
    ax2.set_ylabel('acc')
    l2 = ax2.plot(x, y_acc, 'g', label='acc')
    lines += l2
    # 提取合并的legend
    legs = [l.get_label() for l in lines]     
    # 显示合并的legend
    ax1.legend(lines, legs, loc=0)
    plt.grid()
    plt.show()


def img_inv_transform(img, mean, std, show=True):
    """图片逆变换显示"""
    img = img * std + mean      # denormalize
    img = img.numpy()           # tensor to numpy
    img = img.transpose(1,2,0)  # chw to hwc
    img = img[..., [2,1,0]]     # rgb to bgr
    if show:
        cv2.imshow('raw img', img)
    return img