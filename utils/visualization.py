#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:47:36 2019

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualization(buffer_dict, title='result: '):
    """可视化结果: 至少包含acc(比如验证)
    输入: dict[key, value_list]
            loss(list): [loss1, loss2,..] or [[iter1, loss1], [iter2, loss2], ...]
            acc(list): [acc1, acc2,..] or [[iter1, acc1], [iter2, acc2], ...]
    """
    accs = buffer_dict['acc']
    losses = None
    lrs = None
    if buffer_dict.get('loss', None) is not None:
        losses = buffer_dict['loss']
    if buffer_dict.get('lr', None) is not None:
        lrs = buffer_dict['lr']
        
    if title is None:
        prefix = ""
    else:
        prefix = title
    
    if isinstance(accs[0], list) or isinstance(accs[0], tuple):  # 如果losses列表里边包含idx
        x = np.array(accs)[:,0]
        y_acc = np.array(accs)[:,1]
    else:  # 如果losses列表里边不包含idx只是单纯loss数值
        x = np.arange(len(accs))
        y_acc = np.array(accs)
    # 绘制loss
    prefix += ' accs'
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title(prefix)
    ax1.set_ylabel('acc')
    lines = ax1.plot(x,y_acc, 'r', label='acc')
    # 绘制loss
    if losses is not None and len(losses) > 0:
        if isinstance(losses[0], list) or isinstance(losses[0], tuple):
            x = np.array(losses)[:,0]
            y_loss = np.array(losses)[:,1]
        else:
            x = np.arange(len(losses))
            y_loss = np.array(losses)
        prefix += ' losses'
        ax1.set_title(prefix)
        ax2 = ax1.twinx()
        ax2.set_ylabel('loss')
        l2 = ax2.plot(x, y_loss, 'g', label='loss')
        lines += l2
        
    # 提取合并的legend
    legs = [l.get_label() for l in lines]     
    # 显示合并的legend
    ax1.legend(lines, legs, loc=0)
    plt.grid()
    plt.show()
    
    # 由于量纲问题，lr单独绘制
    if lrs is not None and len(lrs) > 0:
        if isinstance(lrs[0], list) or isinstance(lrs[0], tuple):
            x = np.array(lrs)[:,0]
            y_lr = np.array(lrs)[:,1]
        else:
            x = np.arange(len(lrs))
            y_lr = np.array(lrs)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_title(title + ' lr')
        ax1.set_ylabel('lr')
        lines = ax1.plot(x,y_lr, 'r', label='lr')
        legs = [l.get_label() for l in lines]   
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

