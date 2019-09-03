#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:16:38 2019

@author: ubuntu
"""
import numpy as np
import cv2

def standardize(X):
    """ 标准化到标准正态分布N(0,1): x-mean / std, 每列特征分别做标准化 
    注意：当前standardize跟normalize的说法有混淆的情况，比如batchnorm做的是standardize，但却叫norm
    """
    X_std = X
    mean = X.mean(axis=0)    # 按列求均值
    std = X.std(axis=0)      # 按列求标准差
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]  # 每一列特征单独做自己的标准化(减列均值，除列标准差)
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std

def normalize(X):
    """归一化到[0-1]之间：x / 255"""
    return X / 255


def label_transform(labels, label_transform_dict={1:1, -1:0, 0:0}):
    """默认不改变label的取值范围，但可以通过该函数修改labels的对应范围
    例如svm需要label为[-1,1]，则可修改该函数。
    """   
    if label_transform_dict is None:
            pass
    else:  # 如果指定了标签变换dict
        labels = np.array(labels).reshape(-1)  #确保mat格式会转换成array才能操作
        assert isinstance(label_transform_dict, dict), 'the label_transform_dict should be a dict.' 
        for i, label in enumerate(labels):
            new_label = label_transform_dict[label]
            labels[i] = int(new_label)   # 比如{1:1, 0:-1}就是要把1变为1, 0变为-1
    return labels


def label_to_onehot(labels):
    """标签转换为独热编码：输入的labels需要是从0开始的整数，比如[0,1,2,...]
    输出的独热编码为[[1,0,0,...],
                  [0,1,0,...],
                  [0,0,1,...]]  分别代表0/1/2的独热编码
    """
    assert labels.ndim ==1, 'labels should be 1-dim array.'
    labels = labels.astype(np.int8)
    n_col = int(np.max(labels) + 1)   # 独热编码列数，这里可以额外增加列数，填0即可，默认是最少列数
    one_hot = np.zeros((labels.shape[0], n_col))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot  # (n_samples, n_col)


def onehot_to_label(one_hot_labels):
    """把独热编码变回0-k的数字编码"""
    labels = np.argmax(one_hot_labels, axis=1)  # 提取最大值1所在列即原始从0开始的标签
    return labels


def imresize(img, size, interpolation='bilinear', return_scale=False):
    """把图片img尺寸变换成指定尺寸
    img输入为(h,w,c)这种标准格式
    size输入为(h,w)
    """
    interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4}
    
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        h_scale = size[0] / h
        w_scale = size[1] / w
        return resized_img, w_scale, h_scale


if __name__ == "__main__":
#    labels = np.array([[0,1,0],[0,0,1]])
#    new_labels = onehot_to_label(labels)
    
    img = cv2.imread('./test.jpg')
    cv2.imshow('original', img)
    new_img = imresize(img, (14,14), interpolation='bilinear')
    cv2.imshow('show', new_img)


