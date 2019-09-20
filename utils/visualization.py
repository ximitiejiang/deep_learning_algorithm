#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:47:36 2019

@author: ubuntu
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# %%
def vis_loss_acc(buffer_dict, title='result: '):
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


# %%
def vis_img_bbox(img, bboxes, labels, class_names=None,
        thickness=1, font_scale=0.5):
    """简化版显示img,bboxes,labels
    img
    bboxes
    labels
    """
    from utils.colors import COLORS
    # 准备颜色
    color_list = []
    for color in COLORS.values():
        color_list.append(color)
    color_list.pop(-1) # the last one is white, reserve for text only, not for bboxes
    color_list = color_list * 12  # 循环加长到84，可以显示80类的coco
    random_colors = np.stack(color_list, axis=0)  # (7,3)
#    random_colors = np.tile(random_colors, (12,1))[:len(class_names),:]
    # 开始绘制
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(                  # 画方框
            img, left_top, right_bottom, random_colors[label].tolist(), 
            thickness=thickness)
        label_text = class_names[       # 准备文字
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += ': {:.02f}'.format(bbox[-1])
            
        txt_w, txt_h = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness = 1)[0]
        cv2.rectangle(                  # 画文字底色方框
            img, (bbox_int[0], bbox_int[1]), 
            (bbox_int[0] + txt_w, bbox_int[1] - txt_h - 4), 
            random_colors[label].tolist(), -1)  # -1为填充，正整数为边框thickness
        cv2.putText(
            img, label_text, (bbox_int[0], bbox_int[1] - 2),     # 字体选择cv2.FONT_HERSHEY_DUPLEX, 比cv2.FONT_HERSHEY_COMPLEX好一点
            cv2.FONT_HERSHEY_DUPLEX, font_scale, [255,255,255])
    cv2.imshow('result', img)  


def vis_bbox(bboxes, img=None):
    """绘制一组bboxes(n,4): (xmin,ymin,xmax,ymax)"""
    if img is None:
        img = np.zeros((300, 300)).astype(np.uint8)
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.numpy()
    bboxes_int = bboxes.astype(np.int32)
    for bbox in bboxes_int:
        left_top = (bbox[0], bbox[1])
        right_bottom = (bbox[2], bbox[3])
        cv2.rectangle(img, left_top, right_bottom, (0,255,0), thickness=1)
    cv2.imshow('bboxes', img)
    

# %%
def vis_activation_hist(data_list):
    """用于查看激活层输出值的分布：
    参考：deep learning from scratch， p178
    激活层的输出一般称之为激活值，代表了特征在前向计算过程中是否正常，
    激活值如果集中在左右两侧，则说明有经过激活函数后取值会越来越大，可能产生梯度爆炸或者梯度消失。
    激活值如果集中在中间，则说明激活分布有偏向，在输出表现力上受限，模型学习能力就不够。
    所以激活值应该在+-1之前区域较广泛分布，才是比较合理。
    Args:
        data_list(list): 表示激活函数输出的每一层的值，[d1, d2,..]每个元素为(b,c,h,w)
    """
    plt.figure()
    for i, li in enumerate(data_list):  # 提取每层
        plt.subplot(2, len(data_list)/2+1, i+1)  # 2行
        plt.title(str(i+1)+"-layer")  
        plt.hist(li.flatten(), 30, range=(-3,3))  # 展平成(b*c*h*w,), 然后取30个区间, 由于有bn，所以只统计取值在中间的数。
    plt.show()