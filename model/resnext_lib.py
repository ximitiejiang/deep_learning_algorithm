#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:14:53 2019

@author: ubuntu
"""

import torch.nn as nn
from utils.init_weights import common_init_weights

# %% 基础模块

class Bottleneck(nn.Module):
    """resnext50,101,152及以上的模型使用: 主要是引入group conv组卷积
    
    采用1x1+3x3+1x1的基础结构，并在3x3卷积中进行组卷积。
    1. 为了兼容pytorch的权重，各层命名需要固定为：bn1,bn2,downsample,conv1,conv2, 且不能嵌套在sequential中 
    2. 3x3两边的1x1只用来降低通道数和恢复通道数，所以参数统一为s1,p0, 而3x3只用来过滤特征，既不改变通道数和也不改变尺寸。
    3. 为了跟resnet18兼容，输入的out_channels其实是3x3的输出通道数，而3x3之后的1x1通道数需要再乘以4，identity支路的输出通道也是乘以4
       这也体现了resnet在处理很深的网络时，采用的手段是用1x1先通道降维到跟resnet18一个水平，然后再用1x1通道升维4倍。
    4. 模块全部带有恒等映射，但分3种，一种只有恒等映射，一种加了1x1卷积，但不做下采样，还有一种就是加了1x1卷积，且要做下采样stride=2.
       注意：这里bottleneck跟basicblock有区别，basicblock的layer1没有downsample, 但bottleneck的layer1就有downsample，不过该downsample的stride=1，其他都一样
    """
    def __init__(self, in_channels, out_channels, stride, with_downsample=False, groups=32):
        super().__init__()
        # 针对resnext修改bottleneck部分只需要调整out_channels为width，并落实到3个conv中去。
        if groups == 1: # 则为常规卷积
            width = out_channels   # 参考resnext原论文(width of bottleneck, 即为d)，引入width概念，表示每个分组的通道数
        else:
            width = out_channels * 2  # 这里宽度就是指做组卷积的总宽度
            
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # resnext修改bottleneck的第二个地方：把3x3卷积改为groups conv
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, 
                               padding=1, groups=groups, bias=False)  # 该层用于downsample，需要跟identity分支的downsample一致的stride
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if with_downsample:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * 4))
    
    def forward(self, x):
        identity = x    # 预存输入，作为恒等映射分支的输入
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity  # 恒等映射都要有，区别是有的恒等映射增加了下采样模块，有的没有。
        x = self.relu(x)
        
        return x



# %%
class Resnext(nn.Module):
    """Resnext主模型: 是在resnet模型基础上的结构优化
    结构特点：具有很好的可扩展性，每一个layer下面，都可以无限增加block
    表达方式：resnext50_32x4d代表分为32个组，每个组的通道数是4
    
    结构改变：resnext的每层最终输出通道数跟resnet一样，但输入通道数调整成完全跟前一层输出一致，没有做降维。
    保持了layer之间通道数的单调递增。
    同时引入分组卷积，一方面减少卷积参数，另一方面分组方式比网络加深和加宽都更有效，并且这种分组很容易扩展。
    
    1. 对比resnet34 / resnet50 / resnext50 的基础block:
       如果是basicblock, 
        layer1(2)       layer2(2)       layer3(2)           layer4(2)
        --------------------------------------------------------------------
        64-64           128(s2)-128     256(s2)-256         512(s2)-512
        64-64           128-128         256-256             512-512
                        128-128         256-256             512-512
                        128-128         256-256             512-512
       而如果bottleneck:
        layer1(3)       layer2(4)       layer3(6)           layer4(3)
        --------------------------------------------------------------------
        64-64-256       128-128(s2)-512 256-256(s2)-1024    512-512(s2)-2048
        64-64-256       128-128-512     256-256-1024        512-512-2048
        64-64-256       128-128-512     256-256-1024        512-512-2048
                        128-128-512     256-256-1024
                                        256-256-1024
                                        256-256-1024
        而如果是resnext的bottleneck:
        layer1(3)       layer2(4)       layer3(6)           layer4(3)
        --------------------------------------------------------------------
        128-128-256     256-256(s2)-512 256-256(s2)-1024    512-512(s2)-2048
        128-128-256     256-256-512     256-256-1024        512-512-2048
        128-128-256     256-256-512     256-256-1024        512-512-2048
                        256-256-512     256-256-1024
                                        256-256-1024
                                        256-256-1024
        
    2. 在arch_setting里边增加一个block_expansion参数，该参数表示基于3x3卷积输出之后的通道升维倍数
    """
    # resnext的block个数结构跟resnet是完全一样的
    arch_settings = {
        50: (Bottleneck, 4, (3, 4, 6, 3)),
        101: (Bottleneck, 4, (3, 4, 23, 3)),
        152: (Bottleneck, 4, (3, 8, 36, 3))
    }
    
    def __init__(self, depth, pretrained=None, out_indices=[0,1,2,3], strides=(1, 2, 2, 2),
                 classify_classes=None):
        super().__init__()
        self.pretrained = pretrained
        self.out_indices = out_indices  # 定义输出的layers
        self.strides = strides   # 由于bottleneck的stride跟downsample没有直接相关，存在既有downsample且stride=1的情况，所以这里增加strides作为自定义输入。
        self.classify_classes = classify_classes  # 如果是分类模型则定义分类个数，并设置out_indices=-1就能单输出分类特征
        
        block_class = self.arch_settings[depth][0]      # block类
        block_expansion = self.arch_settings[depth][1]   # 该变量用来指示输出相对于基础输入需要放大的倍数        
        arch_setting = self.arch_settings[depth][2]    #　每个layer的block个数
        # 定义blocks之前的4层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 开始定义所有4个layers
        self.res_layer_names = []
        # 指定初始的输入通道数
        in_channels = 64
        # 外层是layers数
        for i in range(4):
            # 计算每个block的输出通道数： 注意这里basicblock的输出通道数是正常表示，
            # 但bottleneck的输出通道数取的是3x3卷积的输出通道数，而3x3之后的1x1的输出通道数还需要乘以4
            stride = self.strides[i]
            out_channels = 64 * pow(2, i)   # 输出通道(64, 128, 256, 512)
            name = 'layer{}'.format(i + 1)
            layers = []
            # 内层是block数
            for j in range(arch_setting[i]):
                if (i > 0 and j == 0) or (i == 0 and j==0 and block_expansion==4):  # 在basicblock的layer2/3/4的第0层，或者bottleneck的layer0的第0层
                    with_downsample = True   # layer1没有下采样，其他layer的第一层都是下采样
                else:
                    with_downsample = False
                layers.append(block_class(in_channels=in_channels,
                                         out_channels=out_channels,
                                         stride = stride,
                                         with_downsample=with_downsample))
                in_channels = out_channels * block_expansion  # 计算实际的block输出通道数(为3x3输出通道数的4倍)，在block里边也有计算这个倍数。
                stride = 1  # 只对第一个block使用stride=stride, 其他block的stride=1
            self.add_module(name, nn.Sequential(*layers))  # 添加模型，注意：必须添加一个sequential才能保持跟pytorch权值文件一致
            self.res_layer_names.append(name)
        
        # 如果作为分类模型，则增加2层
        if self.classify_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))      #　注意: adaptiveAvgpool用力输出任意w,h的特征图，这里是把w,h收缩为一个点，即w=h=1
            self.fc = nn.Linear(in_channels, self.classify_classes)  # 注意：由于adaptiveAvgpool已经把w,h收缩到1,所以全连接的输入神经元个数就是层数，也就是前面已经乘expansion的结果。不能用out_channel因为没乘expansion.
        # 权重初始化
        self.init_weights()
            
    def forward(self, x):
        # 先执行前面的4个层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 在执行每一个layer,获取需要的layer的输出
        outs = []
        for i, layer_name in enumerate(self.res_layer_names):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        # 如果是分类模型，则执行最后的adaptiveavgpool和fc
        if self.classify_classes is not None:
            x = self.avgpool(x)    # (b, c, 1, 1)
            x = x.reshape(x.shape[0], -1)  # (b, c)
            x = self.fc(x)  # (b, n_classes)
            outs.append(x)
        
        if len(outs) == 1:
            return outs[-1]
        else:            
            return tuple(outs)

    def init_weights(self):
        common_init_weights(self, pretrained=self.pretrained)



# %%    
if __name__ == '__main__':
    import torch
    import numpy as np
    
    name = 'my'
    
    if name == 'ori':
        import torchvision
        from utils.checkpoint import load_checkpoint
        model = torchvision.models.resnext50_32x4d(pretrained=True)
#        load_checkpoint(model, checkpoint_path = '/home/ubuntu/MyWeights/resnet18-5c106cde.pth')
#        load_checkpoint(model, checkpoint_path = '/home/ubuntu/MyWeights/resnet50-19c8e357.pth')
        load_checkpoint(model, checkpoint_path = '/home/ubuntu/MyWeights/resnext50_32x4d-7cdf4587.pth')
        print(model)
    
    if name == 'my':
        model = Resnext(depth=50, 
                        pretrained = '/home/ubuntu/MyWeights/resnext50_32x4d-7cdf4587.pth',
                        classify_classes=10)
        img = np.random.randn(8,3,300,300)
        img = torch.Tensor(img)
        output = model(img)
        