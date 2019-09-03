

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:00:07 2019

@author: ubuntu

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import kaiming_normal_init
from utils.registry_build import registered


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, relu=True, bn=True, 
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class TUM(nn.Module):
    def __init__(self, first_level=True, input_planes=128, is_smooth=True, side_channel=512, scales=6):
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2 * self.input_planes
        self.first_level = first_level
        self.scales = scales
        self.in1 = input_planes + side_channel if not first_level else input_planes

        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)), BasicConv(self.in1, self.planes, 3, 2, 1))
        for i in range(self.scales-2):
            if not i == self.scales - 3:
                self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 2, 1)
                        )
            else:
                self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 1, 0)
                        )
        self.toplayer = nn.Sequential(BasicConv(self.planes, self.planes, 1, 1, 0))
        
        self.latlayer = nn.Sequential()
        for i in range(self.scales-2):
            self.latlayer.add_module(
                    '{}'.format(len(self.latlayer)),
                    BasicConv(self.planes, self.planes, 3, 1, 1)
                    )
        self.latlayer.add_module('{}'.format(len(self.latlayer)),BasicConv(self.in1, self.planes, 3, 1, 1))

        if self.is_smooth:
            smooth = list()
            for i in range(self.scales-1):
                smooth.append(
                        BasicConv(self.planes, self.planes, 1, 1, 0)
                        )
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y, fuse_type='interp'):
        _,_,H,W = y.size()
        if fuse_type=='interp':
            return F.interpolate(x, size=(H,W), mode='nearest') + y
        else:
            raise NotImplementedError
            #return nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x,y],1)
        conved_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)
        
        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(
                    self._upsample_add(
                        deconved_feat[i], self.latlayer[i](conved_feat[len(self.layers)-1-i])
                        )
                    )
        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(
                        self.smooth[i](deconved_feat[i+1])
                        )
            return smoothed_feat
        return deconved_feat


class SFAM(nn.Module):
    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio

        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels,
                                                 self.planes*self.num_levels // 16,
                                                 1, 1, 0)] * self.num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels // 16,
                                                 self.planes*self.num_levels,
                                                 1, 1, 0)] * self.num_scales)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf*_tmp_f)
        return attention_feat


@registered.register_module
class MLFPN(nn.Module):
    """创建Multi Layers Feature Pyramid Net
    1. TUM: 类似与unet/fpn的多级特征融合模块
    """
    def __init__(self, 
                 backbone_type,
                 input_size, 
                 planes, 
                 smooth=True, 
                 num_levels=8, 
                 num_scales=6, 
                 side_channel=512,
                 sfam = False,
                 compress_ratio=16):
        super().__init__()

#        self.phase = phase  # train or test
        self.input_size = input_size    # input img size (512)
        self.planes = planes  # ultimate layers for all tums, default 256, have relation with input size
        self.smooth = smooth  # convs 1x1 are smooth layers
        self.num_levels = num_levels  # how many tums
        self.num_scales = num_scales  # how many scale outputs for each tum
        self.side_channel = side_channel  # use to add to tum input layers
        self.sfam = sfam
        self.compress_ratio = compress_ratio
        
        # build FFM: 
        if backbone_type == 'M2detVGG':
            shallow_in, shallow_out = 512, 256  
            deep_in, deep_out = 1024, 512
        elif backbone_type == 'M2detResnet':
            shallow_in, shallow_out = 512, 256  
            deep_in, deep_out = 1024, 512
            
        self.reduce= BasicConv(
            shallow_in, shallow_out, kernel_size=3, stride=1, padding=1)
        self.up_reduce= BasicConv(
            deep_in, deep_out, kernel_size=1, stride=1)
        
        # build FFM2
        self.leach = nn.ModuleList([
            BasicConv(deep_out + shallow_out, self.planes//2, 
                      kernel_size=(1,1),stride=(1,1))]*self.num_levels)
        
        # build TUM
        tums = []
        for i in range(self.num_levels):
            if i == 0:
                tums.append(
                        TUM(first_level=True, 
                            input_planes=self.planes//2,
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=512))
            else:
                tums.append(
                        TUM(first_level=False,
                            input_planes=self.planes//2,
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=self.planes))
        self.tums = nn.ModuleList(tums)
        
        # build sfam:
        if self.sfam:
            self.sfam_module = SFAM(self.planes, 
                                    self.num_levels, 
                                    self.num_scales, 
                                    compress_ratio=self.compress_ratio)
        # build norm
        self.Norm = nn.BatchNorm2d(256*self.num_levels)

    def init_weights(self):
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        kaiming_normal_init(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0   
        
        for i in range(self.num_levels):
            self.tums[i].apply(weights_init)
        self.reduce.apply(weights_init)
        self.up_reduce.apply(weights_init)
        self.leach.apply(weights_init)
        
    def forward(self, x):
        """Returns the Multi layer output with same scales concated together. [2048] 
        Args:
            x(list): feature list from vgg16, (512, 64, 64), (1024 , 32, 32)
        """
        x_shallow = self.reduce(x[0])  # (b, 256,64,64)
        x_deep = self.up_reduce(x[1])  # (b, 512,32,32)
        base_feature = torch.cat([x_shallow, 
            F.interpolate(x_deep, scale_factor=2, mode='nearest')], 1)  # (b,768,64,64)
        
        tum_outs = [self.tums[0](self.leach[0](base_feature), 'none')]
        for i in range(1, self.num_levels, 1):
            tum_outs.append(self.tums[i](self.leach[i](base_feature), tum_outs[i-1][-1]))
        
        # concate same scale outputs together: tum_outs (8,) -> sources (6,)
        sources = []
        for i in range(self.num_scales, 0, -1):
            sources.append(torch.cat([tum_out[i-1] for tum_out in tum_outs], 1))
        
        if self.sfam:
            sources = self.sfam_module(sources)
        
        sources[0] = self.Norm(sources[0])
        
        return sources
    