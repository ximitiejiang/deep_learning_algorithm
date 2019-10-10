#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:48:09 2019

@author: ubuntu
"""
import torch.nn as nn
import torch.nn.functional as F
from utils.init_weights import common_init_weights

class FCN8sHead(nn.Module):
    """FCN分割模型头
    针对fcn8s输出out_layer=0, fcn16s输出out_layer=1, fcn32s输出out_layer=2
    """
    def __init__(self, 
                 in_channels=(256, 512, 512), 
                 num_classes=21, 
                 featmap_sizes=(60, 30, 15),
                 out_size=480,
                 out_layer=0,
                 upsample_method='interpolate',
                 loss_seg_cfg=None):
        
        super().__init__()
        self.out_layer = out_layer  # 定义最终的输出层(fcn8s, fcn16s, fcn32s的输出层不同)
        self.out_size = out_size
        # 接在最后一层特征的后边
        self.block = nn.Sequential(
                nn.Conv2d(in_channels[-1], 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Conv2d(128, num_classes, kernel_size=1, stride=1))
        # 水平1x1：类似于FPN的literals(1x1)用于统一各个特征图的通道数，便于后续融合
        self.literals = nn.ModuleList([nn.Conv2d(in_channels[0], num_classes, 1),
                                       nn.Conv2d(in_channels[1], num_classes, 1)])
        # 无参上采样：F.interpolate等效于nn.Upsample()
        if upsample_method is None or upsample_method == 'interpolate':
            self.upsample = F.interpolate   
            
        # 带参上采样：nn.ConvTransose2d(), 在FCN中采用带参上采样的转置卷积并不能更提高性能，所以作者把lr=0,相当于无参上采样了。
        # 参考：https://zhuanlan.zhihu.com/p/22976342
        # TODO: 待计算conv transpose参数(从 (21, 60/30/15, 60/30/15) to (21, 480, 480))
        elif upsample_method == 'conv_transpose' or upsample_method == 'dconv':
            self.upsample = nn.ConvTranspose2d()
        
    
    def forward(self, x):
        """从fcnvgg过来x为(3,)
        """        
        # 先计算literals进行通道数统一到21
        l_outs = []
        for i, conv in enumerate(self.literals):
            l_outs.append(conv(x[i]))
        l_outs.append(self.block(x[-1]))
        # 再进行上采样后特征融合
        outs = l_outs
        for i in range(len(outs) - 1, 0, -1): # 从高语义层往低语义层
            outs[i - 1] += F.interpolate(outs[i], scale_factor=2, 
                                         mode='bilinear', align_corners=True)
        # 再进行上采样(转置卷积或者插值法)  
        result = self.upsample(outs[self.out_layer], size=self.out_size, 
                             mode='bilinear', align_corners=True)
        return result  # (b, 21, 480, 480)
    
    
    def init_weights(self):
        common_init_weights(self)
    
    
    def get_losses(self, seg_scores, seg_targets):
        """计算损失: 分割的本质是对每一个像素进行类别预测，所以得到的是score(回归得到的才是preds)
        args:
            seg_scores: (b, 21, 480, 480)
            seg_targets: (b, 480, 480)
        returns:
            loss: item(已缩减为1个值)
        """
        loss = F.cross_entropy(seg_scores, seg_targets)
        result = dict(loss = loss)
        return result
        

# %% fcn16s, fcn32s的模型结构一样，唯一区别在于out_layer, 在config设置好就可以得到不同模型。        
class FCN16sHead(FCN8sHead):
    """fcn16s"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     

class FCN32sHead(FCN8sHead):
    """fcn16s"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)