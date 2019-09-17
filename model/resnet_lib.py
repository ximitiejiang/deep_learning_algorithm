#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:14:53 2019

@author: ubuntu
"""

import torch 
import torch.nn as nn
from utils.init_weights import common_inti_weights

# %% 基础模块

class BasicBlock(nn.Module):
    """resnet18,34及以下的模型使用
    采用3x3+3x3 + residual的结构，类似于vgg的3x3+3x3(模拟出5x5的滤波器核)，但带有residual后就可以叠加更多层。
    
    结构特点：
    1. 为了兼容pytorch的权重，各层命名需要固定为：bn1,bn2,downsample,conv1,conv2, 且不能嵌套在sequential中
    2. 模块第一层conv进行输入输出通道变换，其他层输入输出通道都一样，为输出通道数, 也就是每个basicblock最多让通道数加倍，或者通道数不变。
    3. 所有卷积层的bias都不是默认值，都变为False
    4. 模块全部都带恒等映射分支(residual)，但分2种类型，一种在恒等映射分支中包含conv/bn做下采样，另一种则只是恒等映射，
       所以有下采样的类型的恒等映射分支第一个conv的stride=2，且恒等映射conv1x1的stride=2来实现下采样 
    5. 恒等映射分支的相加点是在batchnorm层之后relu之前，而不是最后的relu之后。
    """
    def __init__(self, in_channels, out_channels, with_downsample=False):
        # 创建residual层，并定义stride
        if with_downsample:
            stride = 2       # 有downsample则stride=2
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d())
        else:
            stride = 1
        # 创建卷积层    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)  # 第一个conv stride跟downsample一样
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False)  # 第二个conv stride永远等于1
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x    # 预存输入，作为恒等映射分支的输入
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity   # 恒等映射都要有，区别是有的恒等映射增加了下采样模块，有的没有。
        x = self.relu(x)
        
        return x


class Bottleneck(nn.Module):
    """resnet50,101,152及以上的模型使用
    采用1x1+3x3+1x1的基础结构(注意，caffe版本的bottleneck跟pytorch版的bottleneck在基础结构上不一样，参考mmdetection的resnet说明)
    这种结构目的是用1x1先通道数降维到跟resnet18一个水平，然后让3x3完成学习，再用1x1升维4倍，本质上是为了把3x3卷积操作的计算量降到跟resnet18一个水平
    结构特点：
    1. 为了兼容pytorch的权重，各层命名需要固定为：bn1,bn2,downsample,conv1,conv2, 且不能嵌套在sequential中 
    2. 3x3两边的1x1只用来降低通道数和恢复通道数，所以参数统一为s1,p0, 而3x3只用来过滤参数，既不改变通道数和也不改变尺寸。
    3. 为了跟resnet18兼容，输入的out_channels其实是3x3的输出通道数，而3x3之后的1x1通道数需要再乘以4，identity支路的输出通道也是乘以4
       这也体现了resnet在处理很深的网络时，采用的手段是用1x1先通道降维到跟resnet18一个水平，然后再用1x1通道升维4倍。
    4. 模块全部带有恒等映射，但分3种，一种只有恒等映射，一种加了1x1卷积，但不做下采样，还有一种就是加了1x1卷积，且要做下采样stride=2
    """
    def __init__(self, in_channels, out_channels, stride, with_downsample=False):
        
        if with_downsample:
            
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * 4))
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False)
        self.bn2
        self.conv3
        self.bn3
        self.relu
    
    def forward(self, x):
        
        return x



# %%
class Resnet(nn.Module):
    """Resnet主模型: 可生成如下5种基础resnet模型
    结构特点：具有很好的可扩展性，每一个layer下面，都可以无限增加block
    可以看到bottleneck的每个3x3(中间那层)的运算通道数数其实跟basicblock一样，这样好处是利用1x1降维很大的节约了3x3卷积的计算量。
    
    1. 如果是basicblock, 
        layer1(2)   layer2(2)   layer3(2)   layer4(2)
        64-64       64-128      128-256     256-512
        64-64       128-128     256-256     512-512
                    128-128     256-256     512-512
                    128-128     256-256     512-512
       而如果bottleneck:
        layer1      layer2      layer3      layer4
        64-64-256   256-128-512 512-256     256-512
        256-64-256  512-128-512 256-256     512-512
        256-64-256  512-128-512 256-256     512-512
                    512-128-512     256-256     512-512
    2. 
    """
    
    arch_settings = {
        18: (BasicBlock, 1, (2, 2, 2, 2)),
        34: (BasicBlock, 1, (3, 4, 6, 3)),
        50: (Bottleneck, 4, (3, 4, 6, 3)),
        101: (Bottleneck, 4, (3, 4, 23, 3)),
        152: (Bottleneck, 4, (3, 8, 36, 3))
    }
    
    def __init__(self, depth, pretrained=None, out_indices=[0,1,2,3], strides=(1, 2, 2, 2)):
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.strides = strides   # 由于bottleneck的stride跟downsample没有直接相关，存在既有downsample且stride=1的情况，所以这里增加strides作为自定义输入。
        block_class = self.arch_settings[depth][0]      # block类
        block_expansion = self.arch_settings[depth][1]   # 该变量用来指示输出相对于基础输入需要放大的倍数        
        arch_setting = self.arch_settings[depth][2]    #　每个layer的block个数

        self.conv1 = nn.Conv2d()
        self.bn1 = nn.BatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d()
        
        self.res_layer_names = []
        # 指定开始的输入通道数
        in_channels = 64
        # 外层是layers数
        for i in range(4):
            # 计算每个block的输出通道数： 注意这里basicblock的输出通道数是正常表示，
            # 但bottleneck的输出通道数取的是3x3卷积的输出通道数，而3x3之后的1x1的输出通道数还需要乘以4
            stride = self.strides[i]
            out_channels = in_channels * pow(2, i)   # 输出通道(64, 128, 256, 512)
            name = 'layer{}'.format(i + 1)
            layers = []
            # 内层是block数
            for j in arch_setting:
                with_downsample = True if j==0 and i > 0 else False  # layer1没有下采样，其他layer的第一层都是下采样
                layers.append(block_class(in_channels=in_channels,
                                         out_channels=out_channels,
                                         stride = stride,
                                         with_downsample=with_downsample))
                in_channels = out_channels * block_expansion  # 计算实际的block输出通道数(为3x3输出通道数的4倍)，在block里边也有计算这个倍数。
                
            self.add_module(name, layers)  # 该方式添加可自定义名称，且添加的list自动转换为sequential
            self.res_layer_names.append(name)
            
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
        # 执行最后的adaptiveavgpool和fc
        
        return tuple(outs)
    
    def init_weights(self):
        common_inti_weights(self, pretrained=self.pretrained)


# %%
class Resnet18(nn.Module):
    arch_settings = {18: (2, 2, 2, 2)}
    def __init__(self, pretrained=None, out_indices=[0,1,2,3]):
        self.out_indices = out_indices
        
        self.conv1
        self.bn1
        self.relu
        self.maxpool
        
        self.res_layer_names = []
        
        in_channels = 64
        for i in range(4):
            out_channels = in_channels * pow(2, i)   # 输出通道(64, 128, 256, 512)
            name = 'layer{}'.format(i + 1)
            layers = []
            for j in self.arch_settings.values():
                with_downsample = True if j==0 and i > 0 else False  # layer1没有下采样，其他layer的第一层都是下采样
                layers.append(BasicBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         with_downsample=with_downsample))
                in_channels = out_channels
            self.add_module(name, layers)  # 该方式添加可自定义名称，且添加的list自动转换为sequential
            self.res_layer_names.append(name)
            
    def forward(self, x):
        # 先执行前面的4个层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 在执行每一个layer,获取需要的layer的输出
        outs = []
        for layer_name in self.res_layer_names:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

# %%    
if __name__ == '__main__':
    
    name = 'my'
    
    if name == 'ori':
        import torchvision
        from utils.checkpoint import load_checkpoint
        model = torchvision.models.resnet50()
    #    load_checkpoint(model, checkpoint_path = '/home/ubuntu/MyWeights/resnet18-5c106cde.pth')
        load_checkpoint(model, checkpoint_path = '/home/ubuntu/MyWeights/resnet50-19c8e357.pth')
        print(model)
    
    if name == 'my':
        model = Resnet(depth=18,
                       pretrained='/home/ubuntu/MyWeights/resnet18-5c106cde.pth')
        