#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:14:53 2019

@author: ubuntu
"""

import torch.nn as nn
from utils.init_weights import common_init_weights

# %% 基础模块

class BasicBlock(nn.Module):
    """resnet18,34及以下的模型使用
    采用3x3+3x3 + residual的结构，类似于vgg的3x3+3x3(模拟出5x5的滤波器核)，但带有residual后就可以叠加更多层。
    
    结构特点：
    1. 为了兼容pytorch的权重，各层命名需要固定为：bn1,bn2,downsample,conv1,conv2, 且不能嵌套在sequential中
    2. 模块第一层conv进行输入输出通道变换，其他层输入输出通道都一样，为输出通道数, 也就是每个basicblock最多让通道数加倍，或者通道数不变。
    3. 所有卷积层的bias都不是默认值，都变为False
    4. 模块全部都带恒等映射分支(residual)，但分2种类型，一种在恒等映射分支中包含conv/bn做下采样，另一种则只是恒等映射，
       对于basicblock来说，layer1没有downsample, 其他layers的第一层都是downsample。
    5. 恒等映射分支的相加点是在batchnorm层之后relu之前，而不是最后的relu之后。
    """
    def __init__(self, in_channels, out_channels, stride, with_downsample=False):
        super().__init__()
        # 创建卷积层    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # 第一个conv stride跟downsample一样
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # 第二个conv stride永远等于1
        self.bn2 = nn.BatchNorm2d(out_channels)
 
        # 创建residual层，并定义stride
        self.downsample = None
        if with_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        
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
    2. 3x3两边的1x1只用来降低通道数和恢复通道数，所以参数统一为s1,p0, 而3x3只用来过滤特征，既不改变通道数和也不改变尺寸。
    3. 为了跟resnet18兼容，输入的out_channels其实是3x3的输出通道数，而3x3之后的1x1通道数需要再乘以4，identity支路的输出通道也是乘以4
       这也体现了resnet在处理很深的网络时，采用的手段是用1x1先通道降维到跟resnet18一个水平，然后再用1x1通道升维4倍。
    4. 模块全部带有恒等映射，但分3种，一种只有恒等映射，一种加了1x1卷积，但不做下采样，还有一种就是加了1x1卷积，且要做下采样stride=2.
       注意：这里bottleneck跟basicblock有区别，basicblock的layer1没有downsample, 但bottleneck的layer1就有downsample，不过该downsample的stride=1，其他都一样
    """
    def __init__(self, in_channels, out_channels, stride, with_downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # 该层用于downsample，需要跟identity分支的downsample一致的stride
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
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
class ResNet(nn.Module):
    """Resnet主模型: 可生成如下5种基础resnet模型
    结构特点：具有很好的可扩展性，每一个layer下面，都可以无限增加block
    可以看到bottleneck的每个3x3(中间那层)的运算通道数数其实跟basicblock一样，这样好处是利用1x1降维很大的节约了3x3卷积的计算量。
    如下，可见在前两层conv上，basicblock和bottleneck是完全一样的通道数。
    而下采样都是在每个layer的第一个block的第一个3x3进行。
    1. 对比：
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
    2. 在arch_setting里边增加一个block_expansion参数，该参数表示基于3x3卷积输出之后的通道升维倍数
    """
    
    arch_settings = {
        18: (BasicBlock, 1, (2, 2, 2, 2)),
        34: (BasicBlock, 1, (3, 4, 6, 3)),
        50: (Bottleneck, 4, (3, 4, 6, 3)),
        101: (Bottleneck, 4, (3, 4, 23, 3)),
        152: (Bottleneck, 4, (3, 8, 36, 3))
    }
    
    def __init__(self, depth, pretrained=None, out_indices=(0, 1, 2, 3), strides=(1, 2, 2, 2),
                 classify_classes=None):
        super().__init__()
        self.pretrained = pretrained
        self.out_indices = out_indices  # 定义输出的layers, 但这里的layer不是层，而是一组block的意思，一共4个大的layers，这里说明每个大layer都要输出
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
        model = torchvision.models.resnet50()
    #    load_checkpoint(model, checkpoint_path = '/home/ubuntu/MyWeights/resnet18-5c106cde.pth')
        load_checkpoint(model, checkpoint_path = '/home/ubuntu/MyWeights/resnet50-19c8e357.pth')
        print(model)
    
    if name == 'my':
        model = Resnet(depth=18, classify_classes=10)
        img = np.random.randn(8,3,300,300)
        img = torch.Tensor(img)
        output = model(img)
        