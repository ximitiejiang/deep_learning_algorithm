#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:17:58 2019

@author: ubuntu
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.activation_lib import activation_dict

def conv_bn_relu(in_channels, out_channels, kernel_size, 
                 with_bn=False, activation='relu', with_maxpool=False, 
                 stride=1, padding=0, ceil_mode=False):
    """卷积1x1 (基于vgg的3x3卷积集成模块)：
    - 可包含n个卷积(2-3个)，但卷积的通道数默认在第一个卷积变化，而中间卷积不变，即默认s=1,p=1(这种设置尺寸能保证尺寸不变)。
      所以只由第一个卷积做通道数修改，只由最后一个池化做尺寸修改。
    - 可包含n个bn
    - 可包含n个激活函数
    - 可包含一个maxpool: 默认maxpool的尺寸为2x2，stride=2，即默认特征输出尺寸缩减1/2
    输出：
        layer(list)
    """
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding))
    # bn
    if with_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    # activation
    if activation is not None:
        activation_class = activation_dict[activation] 
        layers.append(activation_class(inplace=True))
    # maxpool
    if with_maxpool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    return layers

"""
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
"""

class FPN(nn.Module):
    """FPN, 用于对特征进行融合，让浅层(位置信息准确但语义信息少)与深层(位置信息粗略但语义信息丰富)
      的特征进行叠加融合，从而极大提高对小物体的检测效果，对中等大物体的检测效果也有提高。
    FPN的结构分4部分：
    1. 一组1x1：叫lateral_conv，用1x1进行降维统一输出层数为256，统一层数后才能进行接下来的特征叠加。
    2. 累加操作：进行特征融合，把小尺寸特征放大1倍然后跟大尺寸特征直接相加
    3. 一组3x3：叫fpn_conv，用3x3对输出特征进行卷积，消除上采样的混叠效应
    4. 输出：为了增加输出分组，需要增加extra_conv(必须s=2来逐层缩减特征图尺寸)，
       如果有多层extra conv，则第一层extra conv可以指定以FPN的最后一层输入作为extra conv的输入，
       也可以指定FPN的前面最后一层输出作为extra conv的输出。这里默认用输入作为extra conv的输出。
    """
    def __init__(self, 
                 in_channels=(256, 512, 1024, 2048),
                 out_channels=256,
                 use_levels=(1, 2, 3),  # 表示作用在哪几层，默认4层都是，但新的FPN只使用了1,2,3层，0层丢弃
                 num_outs=5,
                 extra_convs_on_inputs=True  # 表示extra输出是把input作为输入，而不是把fpn某一层的输出作为输入(会影响extra conv的输入层数)
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_levels = use_levels
        self.num_outs = num_outs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in self.use_levels:
            lateral_conv = conv_bn_relu(self.in_channels[i], self.out_channels, 1, 
                                        False, None, False, 1, 0)
            fpn_conv = conv_bn_relu(self.out_channels, self.out_channels, 3, 
                                    False, None, False, 1, 1)
            self.lateral_convs.extend(lateral_conv)
            self.fpn_convs.extend(fpn_conv)
        # 增加额外输出
        extra_layers = num_outs - len(self.use_levels) 
        if extra_layers > 0:
            if self.extra_convs_on_inputs:  # FCOS采用的是前一层输出作为extra layer的输入，而之前FPN/Faster RCNN都是采用前一层的输入作为extra layer的输入
                in_channels = self.in_channels[-1]  # 对于多层extra conv，第一层的输入是从input来，其他extra conv就是接前一层extra conv的输出作为输入
            else:
                in_channels = self.out_channels    # 如果extra conv不是作用在输入，则就是作用在输出，则只可能是256的输入层数
            # 添加额外的层数
            for i in range(extra_layers):
                extra_fpn_conv = conv_bn_relu(in_channels, self.out_channels, 3,
                                              False, None, False, 2, 1)  # 注意extra conv需要缩减特征尺寸s=2
                self.fpn_convs.extend(extra_fpn_conv)
                in_channels = self.out_channels
        
            
    def forward(self, x):
        # 计算水平laterals conv
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(x[self.use_levels[i]]))  
        # 计算从小往大叠加top_down
        for i in range(len(laterals) - 1, 0, -1): # 因为是从小往大叠加，所以要逆序
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        # 计算fpn conv
        outs = []
        for i in range(len(laterals)):
            outs.append(self.fpn_convs[i](laterals[i]))   # 有几个lateral输出就计算出几个fpn conv
        # 计算额外输出
        x_in = x[-1]
        for i in range(len(laterals), self.num_outs):
            outs.append(self.fpn_convs[i](x_in))
            x_in = outs[-1]
        
        return tuple(outs)
        

# %%
def conv1x1(inc, outc, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inc, outc, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(outc),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv3x3(inc, outc, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inc, outc, 3, stride, padding=1, bias=False),
        nn.BatchNorm2d(outc),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
    

class FPNSSH(nn.module):
    """带SSH模块的FPN，其中SSH用于进一步做特征融合
    """
    def __init__(self, 
                 in_channels=(64, 128, 256),
                 out_channels=64,
                 use_levels=(0, 1, 2),  # 表示作用在哪几层，默认4层都是，但新的FPN只使用了1,2,3层，0层丢弃
                 num_outs=3):
        super().__init__()
        # 水平变换层：用于统一层数
        self.laterals = []
        for in_channel in in_channels:
            self.laterals.append(conv1x1(in_channel, out_channels, stride=1, leaky=0.1))
        # fpn层：用于去除叠加效应
        self.fpns = []
        for _ in range(len(in_channels)):
            self.fpns.append(conv3x3(out_channels, out_channels, stride=1, leaky=0.1)
        # SSH模块：用于进一步语义融合
        self.conv3x3
        self.conv5x5
        self.conv7x7
    
    
    def forward(self, x):
        outs = []
        for i, lateral in enumerate(self.laterals):
            outs.append(lateral(x[i]))
        for 
        

if __name__ == '__main__':
    model = FPN()
    print(model) 
    
    x1 = torch.randn(2, 256, 200, 272)
    x2 = torch.randn(2, 512, 100, 136)
    x3 = torch.randn(2, 1024, 50, 68)
    x4 = torch.randn(2, 2048, 25, 34) 
    x = (x1, x2, x3, x4)      
    outs = model(x)
"""
FPN(
  (lateral_convs): ModuleList(
    (0): ConvModule(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)))
    (1): ConvModule(
      (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)))
    (2): ConvModule(
      (conv): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1)))
  )
  (fpn_convs): ModuleList(
    (0): ConvModule(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1): ConvModule(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (2): ConvModule(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (3): ConvModule(
      (conv): Conv2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (4): ConvModule(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
  )
)
"""
    
    