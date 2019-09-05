#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:16:38 2019

@author: ubuntu
"""
import numpy as np
import torch
import cv2


# %% 特征相关变换
def standardize(X):
    """ 特征矩阵标准化到标准正态分布N(0,1): x-mean / std, 每列特征分别做标准化 
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
    """特征矩阵归一化到[0-1]之间：x / 255"""
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


# %% 图像相关变换

def imresize(img, size, interpolation='bilinear', return_scale=False):
    """把图片img尺寸变换成指定尺寸，中间会造成宽高比例变化。
    img输入为(h,w,c)这种标准格式
    size输入为(w,h), 注意这里图片尺寸是用w,h而不是h,w(计算机内部一般都用h,w，但描述图片尺寸惯例却是用w,h)
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
        h_scale = size[1] / h
        w_scale = size[0] / w
        return resized_img, w_scale, h_scale


def imrescale(img, scales, interpolation='bilinear', return_scale=False):
    """基于imresize的imrescale，默认的保持比例，用于对图像进行等比例缩放到指定外框的最大尺寸
    注意：一个习惯差异是，计算机内部一般都用h,w，但描述图片尺寸惯例却是用w,h
    比如要求的scales=(1333,800)即w不超过1333，h不超过800
    img: 图片输入格式(h,w,c)
    scales: 比例输入格式(scale_value)或(w, h)
    """
    h, w = img.shape[:2]
    if isinstance(scales, (float, int)): # 如果输入单个数值
        scale_factor = scales
    elif isinstance(scales, tuple): # 如果输入h,w范围
        max_long_edge = max(scales)
        max_short_edge = min(scales)
        long_edge = max(h, w)
        short_edge = min(h, w)
        scale_factor = min(max_long_edge / long_edge, max_short_edge / short_edge)
    
    new_size = (int(w * scale_factor+0.5), int(h * scale_factor+0.5))  # 注意必须是w,h输入
    new_img = imresize(img, new_size, interpolation=interpolation)
    
    if return_scale:
        return new_img, scale_factor
    else:
        return new_img
    

def imflip(img, flip_type='h'):
    """图片翻转：h为水平，v为竖直
    """
    if flip_type=='h':
        return np.flip(img, axis=1)  # 水平翻转
    else:
        return np.flip(img, axis=0)  # 竖直翻转


def imnormalize(img, mean, std):
    """图片的标准化到标准正态分布N(0,1): 每个通道c独立进行标准化操作
    注意：如果是bgr图则mean为(3,)，但如果是gray图则mean为(1,)
    mean (3,)
    std (3,)
    返回img(h,w,c)
    """
    return (img - mean) / std    # (h,w,3)-(3,)/(3,)=(h,w,3)


def imdenormalize(img, mean, std):
    """图片的逆标准化: 每个通道c独立进行操作"""
    return img * std + mean


def bgr2rgb(img):
    return img[..., [2, 1, 0]]


def rgb2bgr(img):
    return img[..., [2, 1, 0]]


def get_dataset_norm_params(dataset):
    """计算数据集的均值和标准差
    输入图片需基于hwc，bgr格式。
    输出: mean, std 也是基于bgr顺序的3个通道的值(3,) (3,)
        
    实例：参考mmcv中cifar10的数据mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]
    以上是先归一化到[0-1]之后再求得均值和方差，本方法所求结果跟该mmcv在std上稍有差异，待澄清。
    """
    all_means = []
    all_stds = []
    for img, _ in dataset: # hwc, bgr
        means = np.mean(img, axis=(0,1)).reshape(1,-1)  #(1,3)
        stds = np.std(img, axis=(0,1)).reshape(1,-1)
        all_means.append(means)
        all_stds.append(stds)
    
    all_means = np.concatenate(all_means, axis=0)
    all_stds = np.concatenate(all_stds, axis=0)
    
    mean = np.mean(all_means, axis=0)
    std = np.mean(all_stds, axis=0)   # 注意这里是求所有图片的平均std, 而不是std的std
    return mean, std
    
    
# %% 变换类
class ImgTransform():
    """常规数据集都是hwc, bgr输出，所以在pytorch中至少需要to_rgb, to_chw, to_tensor
    """
    def __init__(self, mean=None, std=None, to_rgb=None, to_tensor=None, 
                 to_chw=None, flip=None, scale=None, keep_ratio=None):
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
        self.to_tensor = to_tensor
        self.to_chw = to_chw            # 定义转换到chw
        self.scale = scale              # 定义缩放比例
        self.flip = flip                # 定义水平翻转
        self.keep_ratio = keep_ratio    # 定义保持缩放比例
        
    def __call__(self, img):
        """img输入：hwc, bgr"""
        # 默认值
        scale_factor = 1
        # 所有变换
        if self.mean is not None:
            img = imnormalize(img, self.mean, self.std)
        if self.to_rgb:
            img = bgr2rgb(img)
        if self.scale is not None and self.keep_scale: # 如果是固定比例缩放
            img, scale_factor = imrescale(img, self.scale, return_scale=True)
        elif self.scale is not None and not self.keep_scale: #　如果不固定比例缩放
            img, w_scale, h_scale = imresize(img, self.scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32) #
        if self.flip:
            img = imflip(img)
        if self.to_chw:
            img = img.transpose(2, 0, 1) # h,w,c to c,h,w
        ori_shape = img.shape
        
        if self.to_tensor:
            img = torch.tensor(img)
        return (img, ori_shape, scale_factor)


class LabelTransform():

    def __init__(self, to_tensor=None, to_onehot=None):
        self.to_tensor = to_tensor
        self.to_onehot = to_onehot
    
    def __call__(self, label):
        if self.one_hot:
            label = np.array(label)
            label = label_to_onehot(label)
            
        if self.to_tensor:
            label = torch.tensor(label)
            
        return label

    

class BboxTransform():
    """Bbox变换类"""
    def __init__(self):
        pass
    def __call__(self):
        pass


def img_inv_transform(img, mean, std, show=True):
    """图片逆变换显示"""
    img = img * std + mean      # denormalize
    img = img.numpy()           # tensor to numpy
    img = img.transpose(1,2,0)  # chw to hwc
    img = img[..., [2,1,0]]     # rgb to bgr
    if show:
        cv2.imshow('raw img', img)
    return img
        


if __name__ == "__main__":
#    labels = np.array([[0,1,0],[0,0,1]])
#    new_labels = onehot_to_label(labels)
    
    img = cv2.imread('./test.jpg')
    cv2.imshow('original', img)
    new_img = imresize(img, (14,14), interpolation='bilinear')
    cv2.imshow('show', new_img)


