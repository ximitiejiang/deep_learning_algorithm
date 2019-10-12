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
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, interpolation=interp_codes[interpolation])  # (h,w,c)
    if not return_scale:
        return resized_img
    else:
        h_scale = size[1] / h
        w_scale = size[0] / w
        return resized_img, w_scale, h_scale


def imrescale(img, scales, interpolation='bilinear', return_scale=True):
    """基于imresize的imrescale，默认的保持比例，用于对图像进行等比例缩放到指定外框的最大尺寸
    注意：一个习惯差异是，计算机内部一般都用h,w，但描述图片尺寸惯例却是用w,h
    比如要求的scales=(1333,800)即w不超过1333，h不超过800
    img: 图片输入格式(h,w,c)
    scales: 尺度输入格式(scale_value)或(w, h)
    """
    h, w = img.shape[:2]
    if isinstance(scales, (float, int)): # 如果输入单个数值
        scale_factor = scales
    elif isinstance(scales, (tuple,list)): # 如果输入w,h取值范围
        max_long_edge = max(scales)
        max_short_edge = min(scales)
        long_edge = max(h, w)
        short_edge = min(h, w)
        scale_factor = min(max_long_edge / long_edge, max_short_edge / short_edge)
    
    new_size = (int(w * scale_factor+0.5), int(h * scale_factor+0.5))  # 注意必须是w,h输入
    new_img = imresize(img, new_size, interpolation=interpolation)
    
#    show_scale_compare((img.shape[1],img.shape[0]), new_size, scales)
    
    if return_scale:
        return new_img, scale_factor
    else:
        return new_img

def show_scale_compare(old_size, new_size, scale_range):
    """显式scale变换后的图片尺寸对比: 输入(w,h)"""
    img = 255*np.ones((800,800,3))
    cv2.rectangle(img, (0,0), old_size, (0,255,0),2)
    cv2.rectangle(img, (0,0), new_size, (0,0,125),2)
    cv2.rectangle(img, (0,0), scale_range, (0,0,0), 3)
    cv2.imshow('compare', img)


def impad(img, shape, pad_val=0):
    """对图片边缘进行填充: 采用的方式是在图片右下角额外填充。
    Args:
        img (ndarray): 图片.
        shape (tuple): 目标形状.
        pad_val (number or sequence): 填充值.
    Returns:
        pad. 已经填充好的图片
    """
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1], )
    assert len(shape) == len(img.shape)
    
    for i in range(len(shape) - 1):
        assert shape[i] >= img.shape[i]
        
    pad = np.empty(shape, dtype=img.dtype)
    pad[...] = pad_val  # 先生成一个空数组，并全部填充pad_val
    pad[:img.shape[0], :img.shape[1], ...] = img  # 把图片在填入pad数组的左上角，相当于在右下角pad
    return pad


def impad_to_multiple(img, divisor, pad_val=0):
    """对图片边缘进行填充，并确保填充完成的长和宽能够被某数整除。
    用于神经网络的多尺度图片输入。
    Args:
        img (ndarray): 原始图片.
        divisor (int): 图片w,h能被整除的数字
        pad_val (number or sequence): 填充值.
    Returns:
        pad. 已经填充好的图片
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor  # 获得可以整除的h
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor  # 获得可以整除的w
    return impad(img, (pad_h, pad_w), pad_val)   # 填充
    

def imflip(img, flip_type='h'):
    """图片翻转：h为水平，v为竖直
    """
    if flip_type == 'h':
        return np.flip(img, axis=1)  # 水平翻转
    elif flip_type == 'v':
        return np.flip(img, axis=0)  # 竖直翻转


def bbox_flip(bboxes, img_shape, flip_type='h'):
    """bbox翻转: 这是自己实现的一个版本，可以同时支持bbox的水平和垂直翻转
    Args:
        bboxes(list): [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
        img_shape(tuple): (h, w)
    Returns:
        fliped_img(array): (h,w,c)
    """
    assert flip_type in ['h','v', 'horizontal', 'vertical']
    bboxes=np.array(bboxes)
    h, w = img_shape[0], img_shape[1]
    assert bboxes.shape[-1] == 4
    
    if flip_type == 'h':
        flipped = bboxes.copy()
        # xmin = w-xmax-1, xmax = w-xmin-1
        flipped[...,0] = w - bboxes[..., 2] - 1
        flipped[...,2] = w - bboxes[..., 0] - 1
    elif flip_type == 'v':
        flipped = bboxes.copy()
        flipped[...,1] = h - bboxes[..., 3] - 1
        flipped[...,3] = h - bboxes[..., 1] - 1
        
    return flipped


def imnormalize(img, mean, std):
    """图片的标准化到标准正态分布N(0,1): 每个通道c独立进行标准化操作
    注意：如果是bgr图则mean为(3,)，但如果是gray图则mean为(1,)
    mean (3,)
    std (3,)
    返回img(h,w,c)
    """
    img = img.astype(np.float32)  # 为避免uint8与float的计算冲突，在计算类transform都增加类型转换
    mean = np.array(mean).reshape(-1)
    std = np.array(std).reshape(-1)
    img = (img - mean) / std    # (h,w,3)-(3,)/(3,)=(h,w,3)
    return img.astype(np.float32)


def imdenormalize(img, mean, std):
    """图片的逆标准化: 每个通道c独立进行操作
    img(h,w,c)
    mean(3,)
    std(3,)
    """
    return img * std + mean   # (h,w,3) *(3,) +(3,)=(h,w,3)


def bgr2rgb(img):
    return img[..., [2, 1, 0]]


def rgb2bgr(img):
    return img[..., [2, 1, 0]]


def to_tensor(data):
    """pytorch专用转换成tensor的自定义函数：
    确保img被转换为float32即FloatTensor(跟weight匹配)，int被转化为int64即LongTensor(跟交叉熵公式匹配)
    """
    if isinstance(data, (int, np.int64, np.int32, np.int8)):  # python3中只有一种整数类型int, 没有long的类型
        return torch.LongTensor([data])
    if isinstance(data, float): # python3中只有一种浮点数类型float，没有其他类型
        return torch.FloatTensor([data])
    
    if isinstance(data, np.ndarray):  # img numpy to tensor
        return torch.from_numpy(data)
    
    if isinstance(data, torch.Tensor): # tensor to tensor
        return data
    if isinstance(data, list):   # list to tensor (注意字符串不能转tensor)
        return torch.tensor(data)
    else:
        raise TypeError('not recognized data type for to_tensor.')


def to_device(data, device):
    """用于把数据送入device: 可一次性把一个list中的tensor都送入同一device"""
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    if isinstance(data, list) and isinstance(data[0], torch.Tensor):
        data = [data[i].to(device) for i in range(len(data))]
    return data
        

def get_dataset_norm_params(dataset):
    """计算数据集的均值和标准差
    输入图片需基于hwc，bgr格式。
    输出: mean, std 也是基于bgr顺序的3个通道的值(3,) (3,)
        
    实例：参考mmcv中cifar10的数据mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]
    以上是先归一化到[0-1]之后再求得均值和方差，本方法所求结果跟该mmcv一致。 
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


from utils.colors import get_pallete    
def label2color(img, pallete='voc'):
    """用于在语义分割中把预测的像素标签转换为图片颜色
    ags:
        img: (h, w), 其中的每个像素值为0~20,代表某一类别。
        pallete: (m, 3)
    return
        new_img: hwc, bgr, uint8(uint8才能被cv2正确显示)
    """
    colors = get_pallete(pallete)
    h, w = img.shape
    new_img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            new_img[i, j, :] = colors[img[i, j]][[2,1,0]] # rgb to bgr
    return new_img.astype(np.uint8)  # 采用hwc, bgr，便于cv2显示



    
# %% 变换类
class AugTransform():
    """数据增强变换"""
    def __init__(self):
        pass

class ImgTransform():
    """常规数据集都是hwc, bgr输出，但pytorch操作是在chw,rgb条件下进行。
    所以在pytorch中至少需要to_rgb, to_chw, to_tensor
    args:
        mean, std: 标准化，[B, G, R]均值，[B, G, R]标准差
        norm: 是否归一化
        to_rgb: 转rgb
        to_tensor: 转tensor
        to_chw: hwc转chw
        flip_ratio: 水平翻转比例
        scale: 图片目标尺寸(w, h)
        size_divisor: 边缘填充目标除数
        keep_ratio: 缩放是否保持比例
    """
    def __init__(self, mean=None, std=None, norm=None, to_rgb=None, to_tensor=None, 
                 to_chw=None, flip_ratio=None, scale=None, size_divisor=None, keep_ratio=None):
        self.mean = mean
        self.std = std
        self.norm = norm
        self.to_rgb = to_rgb
        self.to_tensor = to_tensor
        self.to_chw = to_chw            # 定义转换到chw
        self.flip_ratio = flip_ratio    # 定义水平翻转
        self.scale = scale              # 定义缩放后的尺寸，比如[1300, 800]
        self.size_divisor = size_divisor
        self.keep_ratio = keep_ratio    # 定义保持缩放比例
        
    def __call__(self, img):
        """img输入：hwc, bgr
           img输出：chw, rgb, tensor
        """
        ori_shape = img.shape  #(h,w,c)
        # 默认值
        scale_factor = [1, 1, 1, 1] # 假定
        # 所有变换
        if self.to_rgb:
            img = bgr2rgb(img)
        if self.mean is not None and self.norm:  # 标准化+归一化
            img = img / 255
            img = imnormalize(img, self.mean, self.std)
        if self.mean is not None and not self.norm:  # 标准化, 放在bgr2rgb之后做，从而提供的mean也必须是rgb顺序
            img = imnormalize(img, self.mean, self.std)
        if self.scale is not None and self.keep_ratio: # 如果是固定比例缩放
            img, scale_factor = imrescale(img, self.scale, return_scale=True)
        elif self.scale is not None and not self.keep_ratio: #　如果不固定比例缩放
            img, w_scale, h_scale = imresize(img, self.scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32) #变成4个scale目的是提供bbox的xmin/ymin/xmax/ymax的缩放比例
        scale_shape = img.shape
        
        flip = False
        if self.flip_ratio is not None and self.flip_ratio > 0:
            flip = True if np.random.rand() < self.flip_ratio else False  # 随机一个均匀分布(0-1)的值
            if flip:
                img = imflip(img)
                
        if self.size_divisor is not None:
            img = impad_to_multiple(img, self.size_divisor)
        pad_shape = img.shape  # (h,w,c)
        
        if self.to_chw:
            img = img.transpose(2, 0, 1) # h,w,c to c,h,w，注意这里用transpose是numpy的命令所以能用在3d数据，但如果是tensor数据就会报错，因为tensor的transpose只支持2d, permute才支持3d
            img = np.ascontiguousarray(img) # numpy在transpose之后可能导致not contiguous问题产生报错，参考https://discuss.pytorch.org/t/negative-strides-of-numpy-array-with-torch-dataloader/28769
            
        if self.to_tensor:
            img = to_tensor(img)
        return (img, ori_shape, scale_shape, pad_shape, scale_factor, flip)


class LabelTransform():
    """标签变换：通常不受别的变换影响"""
    def __init__(self, to_tensor=None, to_onehot=None):
        self.to_tensor = to_tensor
        self.to_onehot = to_onehot
    
    def __call__(self, label):
        if self.to_onehot:
            label = np.array(label)
            label = label_to_onehot(label)
            
        if self.to_tensor:
            label = to_tensor(label)
            
        return label
    
    
class BboxTransform():
    """Bbox变换类: bboxes(m, 4), 这个变换比较特殊，需要基于img_transform的结果进行变换。
    所以初始化时不输入什么参数，而在call的时候输入所有变换参数。
    注意：bbox的变换顺序需要跟img一致，由于img是先scale再flip，所以bbox也采用先scale再flip
    因此输入用于flip的shape也应该是scale之后的shape
    """
    def __init__(self, to_tensor=None):
        self.to_tensor = to_tensor
        
    def __call__(self, bboxes, img_shape, scale_factor, flip):
        gt_bboxes = bboxes * scale_factor   # (m, 4) * (4,) -> (m, 4)
        
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape, flip_type='h')
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1])
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0])
        
        if self.to_tensor:
            gt_bboxes = to_tensor(gt_bboxes)
        return gt_bboxes


class SegTransform():
    """对语义分割图semantic segmentation(一般是png图片)进行预处理：seg(h, w, 3), 主要收到scale, flip, pad的影响
    """

    def __init__(self, to_tensor=None, scale=None, keep_ratio=None, size_divisor=None, seg_scale_factor=None):
        self.to_tensor = to_tensor
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.size_divisor = size_divisor
        self.seg_scale_factor = seg_scale_factor

    def __call__(self, img, flip):
        # 缩放
        if self.scale is not None and self.keep_ratio: # 如果是固定比例缩放
            img = imrescale(img, self.scale, interpolation='nearest', return_scale=False)
        elif self.scale is not None and not self.keep_ratio: #　如果不固定比例缩放
            img = imresize(img, self.scale, interpolation='nearest', return_scale=False)
        # 翻转
        if flip:
            img = imflip(img)
        # pad
        if self.size_divisor is not None:
            img = impad_to_multiple(img, self.size_divisor)
        # 忽略标签中的255，把他用0代替，作为背景
        img[img == 255] = 0
        # 额外一个seg scale
        if self.seg_scale_factor is not None:
            img = imrescale(img, self.seg_scale_factor, interpolation='nearest')
        # tensor
        if self.to_tensor:
            img = to_tensor(img)
        return img


class MaskTransform():
    """对mask进行变化。
    注意：mask跟segment map的区别，两者都是用来做分割任务的训练数据。
    其中segment map是voc中用来做语义分割或者实例分割的png完整图片。
    而mask是coco中用来做分割的一组数据，并不是图片。
    """
    def __init__(self, to_tensor):
        self.to_tensor = to_tensor
        
    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        # aspect ratio unchanged
        if isinstance(scale_factor, float):
            masks = [
                imrescale(mask, scale_factor, interpolation='nearest')
                for mask in masks
            ]
        # aspect ratio changed
        else:
            w_ratio, h_ratio = scale_factor[:2]
            if masks:
                h, w = masks[0].shape[:2]
                new_h = int(np.round(h * h_ratio))
                new_w = int(np.round(w * w_ratio))
                new_size = (new_w, new_h)
                masks = [
                    imresize(mask, new_size, interpolation='nearest')
                    for mask in masks
                ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks

    
"""
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose)

class AugmentTransform():
    '''数据增强变换类：通过封装albumentations来实现
    参考：https://github.com/albu/albumentations, 
    实例：https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
    实例：https://github.com/albu/albumentations/blob/master/notebooks/example_16_bit_tiff.ipynb
    '''
    def __init__(self, p=0.5,
                 horizontal_flip=False,
                 random_rotate_90=False):
        aug_list = []
        if horizontal_flip:
            aug_list.append(HorizontalFlip())
        if random_rotate_90:
            aug_list.append(RandomRotate90())
        self.aug = Compose(aug_list)
        
    def __call__(self, img, label, bbox):
        auged = self.aug()
        
        return auged['image'], auged['']
"""


from utils.visualization import vis_img_bbox
from utils.tools import get_time_str
def transform_inv(img, bboxes=None, labels=None, mean=None, std=None, class_names=None,show=False,save=None):
    """图片和bbox的逆变换和显示，为了简化处理，不做scale/flip的逆变换，这样便于跟bbox统一比较
    注意，逆变换过程需要注意的地方很多，尽可能用这个函数完成。
    args:
        img: (c,h,w), 可以是img or segmap
        bboxes: (n,4)
        labels: (n)
    """
    # 为了便于计算时广播，变换成一维array
    if mean is not None:
        mean = np.array(mean).reshape(-1)
        std = np.array(std).reshape(-1)
    # tensor to array, to cpu
    if isinstance(img, torch.Tensor):
        img = img.cpu()
        img = img.numpy()
    if bboxes is not None:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu()
            labels = labels.cpu()
        bboxes = bboxes.numpy()
        labels = labels.numpy()
    # chw to hwc
    img = img.transpose(1,2,0)
    img = np.ascontiguousarray(img)
    # rgb to bgr
    img = img[..., [2,1,0]]     # 该步必须在hwc之后操作
    # denormalize，该步必须在bgr之后做，因为mean/std的顺序是BGR顺序 (512,512,3) * (3,) + (3,)
    img = imdenormalize(img, mean, std)  
    # 最后截取0-255的无符号整数
    img = np.clip(img, 0, 255).astype(np.uint8)  # 只有uint8的数据格式才能被opencv正确显示
    if show:
        if bboxes is None:  # 只显示img
            cv2.imshow('raw img', img)  # hwc, bgr
        else:   # 同时显示img,bboxes, labels
            vis_img_bbox(img, bboxes, labels, class_names)
    if save is not None:
        name = save + get_time_str() + '.jpg'
        cv2.imwrite(name, img)
    return img, bboxes, labels
        

# %%
if __name__ == "__main__":
#    labels = np.array([[0,1,0],[0,0,1]])
#    new_labels = onehot_to_label(labels)
    
    img = cv2.imread('../example/2.jpg')
    cv2.imshow('original', img)
    new_img, scale_factor = imrescale(img, (700,600), interpolation='bilinear', return_scale=True)
    cv2.imshow('show', new_img)


