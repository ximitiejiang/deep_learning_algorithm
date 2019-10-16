#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:39:44 2019

@author: ubuntu
"""
import torch.nn as nn
from utils.init_weights import common_init_weights

def conv3x3(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
    
def conv3x3_1x1(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),   # 注意，这里带了一个组卷积
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )

class MobileNetV1(nn.Module):
    """mobilenetv1的简化版本，权重来自https://github.com/biubug6/Pytorch_Retinaface
    未确认是否能用别的版本的权重。
    结构上采用1x1+3x3的模块
    """
    def __init__(self, 
                 pretrained=None,
                 out_stages=(1, 2, 3)):
        super().__init__()
        self.pretrained = pretrained
        self.out_stages = out_stages
        
        self.stage1 = nn.Sequential(
            conv3x3(3, 8, 2, leaky = 0.1),    # 3
            conv3x3_1x1(8, 16, 1),   # 7
            conv3x3_1x1(16, 32, 2),  # 11
            conv3x3_1x1(32, 32, 1),  # 19
            conv3x3_1x1(32, 64, 2),  # 27
            conv3x3_1x1(64, 64, 1),  # 43
        )
        
        self.stage2 = nn.Sequential(
            conv3x3_1x1(64, 128, 2),  # 43 + 16 = 59
            conv3x3_1x1(128, 128, 1), # 59 + 32 = 91
            conv3x3_1x1(128, 128, 1), # 91 + 32 = 123
            conv3x3_1x1(128, 128, 1), # 123 + 32 = 155
            conv3x3_1x1(128, 128, 1), # 155 + 32 = 187
            conv3x3_1x1(128, 128, 1), # 187 + 32 = 219
        )
        
        self.stage3 = nn.Sequential(
            conv3x3_1x1(128, 256, 2), # 219 +3 2 = 241
            conv3x3_1x1(256, 256, 1), # 241 + 64 = 301
        )
        
#        self.avg = nn.AdaptiveAvgPool2d((1,1))
#        self.fc = nn.Linear(256, 1000)
        
    def forward(self, x):
        outs = []
        x = self.stage1(x)
        outs.append(x)
        x = self.stage2(x)
        outs.append(x)
        x = self.stage3(x)
        outs.append(x)
        outs = [outs[i-1] for i in self.out_stages]
        return outs
    
    
    def init_weights(self):
        common_init_weights(self, self.pretrained)


if __name__ == "__main__":
    import torch
    model = MobileNetV1(pretrained = '/media/ubuntu/4430C54630C53FA2/SuLiang/MyWeights/retinaface_backbone/mobilenetV1X0.25_pretrain.tar')
    model.init_weights()
    img = torch.randn(4, 3, 320, 320)
    outs = model(img)