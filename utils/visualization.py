#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:47:36 2019

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


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

