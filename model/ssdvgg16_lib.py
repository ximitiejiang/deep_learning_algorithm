#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:54:44 2019

@author: ubuntu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.checkpoint import load_checkpoint
from utils.init_weights import common_init_weights, kaiming_init, constant_init, normal_init, xavier_init
from model.activation_lib import activation_dict

# %% 最简版ssd vgg16

def vgg3x3(num_convs, in_channels, out_channels, with_bn=False, activation='relu', with_maxpool=True, 
            stride=1, padding=1, ceil_mode=True):
    """vgg的3x3卷积集成模块：
    - 可包含n个卷积(2-3个)，但卷积的通道数默认在第一个卷积变化，而中间卷积不变，即默认s=1,p=1(这种设置尺寸能保证尺寸不变)。
      所以只由第一个卷积做通道数修改，只由最后一个池化做尺寸修改。
    - 可包含n个bn
    - 可包含n个激活函数
    - 可包含一个maxpool: 默认maxpool的尺寸为2x2，stride=2，即默认特征输出尺寸缩减1/2
    输出：
        layer(list)
    """
    layers = []
    for i in range(num_convs):
        # conv3x3
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=padding))
        # bn
        if with_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        # activation
        activation_class = activation_dict[activation] 
        layers.append(activation_class(inplace=True))
        in_channels = out_channels
    # maxpool
    if with_maxpool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    return layers


class SSDVGG16(nn.Module):
    """ vgg16是指带参层有16层(13conv + 3linear)
    vgg的最大贡献在于：提出了2个3x3卷积相当于一个5x5卷积的感受野，3个3x3的卷积相当于一个7x7卷积的感受野。
    因此vgg都是通过2个3x3卷积和3个3x3卷积的组合。
    
    ssdvgg16是在基础版VGG16结构上ssd修改部分包括：
    去掉最后一层maxpool然后增加一层maxpool，增加extra convs, l2norm
              img               (3,  h, w)
        ----------------------------------------
              3x3               (64, h, w)
              3x3               (64, h, w)
              maxpool2x2    s2  (64, h/2, w/2)
              3x3               (128,h/2, w/2)
              3x3               (128,h/2, w/2)
              maxpool2x2    s2  (128, h/4, w/4)
              3x3               (256, h/4, w/4)
              3x3               (256, h/4, w/4)
              3x3               (256, h/4, w/4)
              maxpool2x2    s2  (256, h/8, w/8)
              3x3               (512, h/8, w/8)
              3x3               (512, h/8, w/8)
              3x3               (512, h/8, w/8)
              maxpool2x2    s2  (512, h/16, w/16)
              3x3               (512, h/16, w/16)
              3x3               (512, h/16, w/16)
              3x3               (512, h/16, w/16)
        add   maxpool2x2    s2  (512, h/16, w/16)
        ------------------------------------------
        add
              3x3(p=6,d=6)  (1024, )
              1x1           (1024, )
        ------------------------------------------
        extra
              1x1           ()
              3x3
              1x1
              3x3
              1x1
              3x3
              1x1
              3x3
    """
    arch_setting = {16: [2,2,3,3,3]}  # 16表示vgg16，后边list表示有5个blocks，每个blocks的卷积层数
    
    def __init__(self, 
                 num_classes=2,
                 pretrained=None,
                 out_feature_indices=(22,34),
                 extra_out_feature_indices = (1, 3, 5, 7),
                 l2_norm_scale=20.):
        super().__init__()
        self.blocks = self.arch_setting[16]
        self.num_classes = num_classes
        self.out_feature_indices = out_feature_indices
        self.extra_out_feature_indices = extra_out_feature_indices
        self.l2_norm_scale = l2_norm_scale
        
        #构建所有vgg基础层
        vgg_layers = []
        in_channels = 3
        for i, convs in enumerate(self.blocks):
            out_channels = [64, 128, 256, 512, 512] # 输出通道数
            block_layers = vgg3x3(convs, in_channels, out_channels[i])
            vgg_layers.extend(block_layers) # 用extend而不是append
            in_channels = out_channels[i]
            
        vgg_layers.pop(-1) # 去掉最后一层max pool
        self.features = nn.Sequential(*vgg_layers) 
        
        # ssd额外添加maxpool + 2层conv
        # 注意命名需要跟前面层一致，才能确保加载权重是正确的。
        self.features.add_module(
                str(len(self.features)), nn.MaxPool2d(kernel_size=3, stride=1, padding=1))# 最后一个maxpool的stride改为1
        self.features.add_module(
                str(len(self.features)), nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)) # 空洞卷积
        self.features.add_module(
                str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
                str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
                str(len(self.features)), nn.ReLU(inplace=True))
        
        # 构建ssd额外卷积层和l2norm层(ssd论文提及)
        self.extra = self.make_extra_block(in_channels=1024)
        self.l2_norm = L2Norm(self.features[out_feature_indices[0] - 1].out_channels, l2_norm_scale)


     
    def make_extra_block(self, in_channels):
        """额外增加10个conv，用来获得额外的更多尺度输出
        extra_setting = {300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256)}
        """
        layers = []
        layers.append(nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0))
        layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)) # s=2
        
        #layers.append(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0))  # 去除
        layers.append(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0))
        
        #layers.append(nn.Conv2d(128, 128, kernel_size=1, stride=2, padding=1))  # s=2
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        
        layers.append(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0))
        layers.append(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0))
        return nn.Sequential(*layers)
    
    def init_weight(self, pretrained=None):
        """用于模型初始化，统一在detector中进行"""
        # 载入vgg16_caffe的权重初始化vgg
        common_init_weights(self.features, pretrained)
        
#        if pretrained is not None:
#            load_checkpoint(self, pretrained, map_location=None)
#        # 其他新增的features层的初始化
#        else:
#            for m in self.features.modules():
#                if isinstance(m, nn.Conv2d):
#                    kaiming_init(m)
#                elif isinstance(m, nn.BatchNorm2d):
#                    constant_init(m, 1)
#                elif isinstance(m, nn.Linear):
#                    normal_init(m, std=0.01)
        # exra层初始化
        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        # l2 norm层初始化
        constant_init(self.l2_norm, self.l2_norm.scale)
    
    def forward(self, x):
        outs = []
        # 前向计算features层
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)
        # 前向计算extra层
        extra_out_feature_indices = (1, 3, 5, 7)
        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            if i in extra_out_feature_indices:
                outs.append(x)
        # 前向计算l2 norm
        outs[0] = self.l2_norm(outs[0])
        return tuple(outs)
        
        
class L2Norm(nn.Module):
    """l2正则化： x^2"""
    def __init__(self, n_dims, scale=20., eps=1e-10):
        super().__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale
    
    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt() + self.eps  #  l2 = sqrt(sum(xi^2))
        return self.weight[None, :, None].expand_as(x) * x / norm


if __name__ == "__main__":
    import numpy as np
    model = SSDVGG16()
    print(model)
    img = np.ones((8, 3, 300, 300))  # b,c,h,w
    img = torch.tensor(img).float()
    out = model(img)