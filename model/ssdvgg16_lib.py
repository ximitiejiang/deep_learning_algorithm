#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:54:44 2019

@author: ubuntu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_weights import common_init_weights, constant_init, xavier_init
from model.activation_lib import activation_dict
from utils.visualization import vis_activation_hist

# %% 最简版ssd vgg16

def vgg3x3(num_convs, in_channels, out_channels, with_bn=False, activation='relu', with_maxpool=True, 
            stride=1, padding=1, ceil_mode=False):
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
    从数字图像处理角度，5x5, 7x7的滤波器是主流，所以，vgg为了获得5x5,7x7的滤波器，并没有像alexnet那样直接
    用5x5,7x7的卷积核，而是提出了2个3x3卷积相当于一个5x5卷积的感受野，3个3x3的卷积相当于一个7x7卷积的感受野。
    因此vgg都是通过2个3x3卷积和3个3x3卷积的组合，这样的好处是既有足够大的感受野，又节省了参数，还增加模型非线性能力。
    
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
              1x1           
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
                 pretrained=None,
                 out_feature_indices=(22,34),
                 extra_out_feature_indices = (1, 3, 5, 7),
                 l2_norm_scale=20.,
                 classify_classes=None):
        super().__init__()
        self.blocks = self.arch_setting[16]
        self.out_feature_indices = out_feature_indices
        self.extra_out_feature_indices = extra_out_feature_indices
        self.l2_norm_scale = l2_norm_scale
        self.classify_classes = classify_classes  # 如果不为None，则用来做分类器，增加几层进行分类
        self.pretrained = pretrained
        
        #构建所有vgg基础层
        vgg_layers = []
        in_channels = 3
        for i, convs in enumerate(self.blocks):
            out_channels = [64, 128, 256, 512, 512] # 输出通道数
            block_layers = vgg3x3(convs, in_channels, out_channels[i], ceil_mode=True)  # 注意前4个maxpool的ceil_mode=True，所以最后第5个maxpool被弹出丢弃
            vgg_layers.extend(block_layers) # 用extend而不是append
            in_channels = out_channels[i]
            
        vgg_layers.pop(-1) # 去掉最后一层max pool(改为ceil mode=False默认设置的1x1池化)
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
        self.l2_norm = L2Norm(self.features[out_feature_indices[0] - 1].out_channels, l2_norm_scale) # 维度是第22层的输出通道数(512)
        
        # 额外增加2层用来做分类模型(自适应平均池化+全连接，参考resnet)
        if self.classify_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
            self.fc = nn.Linear(1024, self.classify_classes)

     
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
    
    def init_weights(self):
        """用于模型初始化，统一在detector中进行"""
        # 载入vgg16_caffe的权重初始化vgg
        common_init_weights(self, pretrained=self.pretrained)        
        # exra层初始化
        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        # l2 norm层初始化
        constant_init(self.l2_norm, self.l2_norm.scale)
        # 如果作为分类模型的初始化补充
        if self.classify_classes is not None:
            common_init_weights(self.fc)
    
    def forward(self, x):
        outs = []
#        hist_list = []  # 检查输出的分布情况
        # 前向计算features层
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)
#                hist_list.append(x.detach().numpy())
        # 前向计算extra层
        extra_out_feature_indices = (1, 3, 5, 7)
        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            if i in extra_out_feature_indices:
                outs.append(x)
#                hist_list.append(x.detach().numpy())
        
        # 前向计算l2 norm
        outs[0] = self.l2_norm(outs[0])  # 只计算第一个尺寸的特征图
        
        # 加上l2norm之后的数据，统一绘制: 第一张和最后一张是l2norm前后对比。
#        hist_list.append(outs[0].detach().numpy())
#        vis_activation_hist(hist_list)
        # 如果作为分类器，则只输出一个
        if self.classify_classes is not None:
            x = self.avgpool(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc(x)
            outs.append(x)
        
        if len(outs) == 1:
            return outs[-1]
        else:
            return tuple(outs)
        
        
class L2Norm(nn.Module):
    """l2归一化层： 该技术来自ParseNet，主要出发点是因为多层feature map的激活值大小分布差距比较大，
    如果没有经过 norm，会导致激活值普遍较大的 feature map 对融合后的结果影响更大，所以要在融合之前做l2norm。
    ssd在vgg输出的第一个特征图(38x38)后边增加了一个L2Norm，
    原因是该层比较靠前norm比较大，因而采用L2norm对特征图进行归一化，保证他跟后边高语义特征的层的差异不会太大。
    注意：l2norm跟batchnorm的区别，batchnorm是在batch这个轴求均值，也就是把一个batch(b,c,h,w)得到一个均值mean(c,h,w)
    而l2norm求均方值是在channel通道这个轴求，也就是把一个batch(b,c,h,w)得到norm(b, 1, h, w)
        公式是y = weight * xi/sqrt(sum(x^2)), 其中weight是一个可学习参数，mmdetection里边把他用通道个数个参数值作为参数。
    也就是说先求出一个总的通道的norm值(b,1,h,w)，然后通过学习来决定每个通道的权重，一共512个通道，也就有512个权重值。
    
    """
    def __init__(self, n_dims, scale=20., eps=1e-10):
        super().__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))  # (512,) 定义一个权重信息在公式中，并把这个权重值作为可学习参数放入nn.Module的paramters中去，从而能够在反向传播中自动更新。
        self.eps = eps
        self.scale = scale
    
    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt() + self.eps  # 求和是沿着通道方向axis=1来求和，得到(b, 1, 38, 38)
        return self.weight[None, :, None, None].expand_as(x) * x / norm  #  (1, 512, 1, 1)*(b,c,h,w) /(b,1,h,w) -> ()


if __name__ == "__main__":
    import numpy as np   
    # 检查conv计算w,h的取整方式：maxpool可以手动设置，但conv不能，其默认方式是下取整
    img = np.ones((8,3,9,9))
    img = torch.tensor(img).float()
    conv = nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1)
    out = conv(img)  # (9-3+2)/2 +1 = 9
    # 下取整计算
    img = np.ones((8,3,10,10))
    img = torch.tensor(img).float()
    conv = nn.Conv2d(3,64,kernel_size=3, stride=2, padding=1)
    out2 = conv(img)  # (10-3+2)/2 +1 = 5 (默认下取整，跟maxpool一样，只不过maxpool可以手动指定ceil mode，但conv不能手动指定)
    
    # 检查ssdvgg的输出是否正确
    """ 这个batch的数据无法计算下去，主要是因为：padding之后变成(4, 3, 236, 300)
    但mmdetection是从ori (375, 500, 3)变成img_shape (300, 300, 3)
    
    [{'ori_shape': (375, 500, 3), 'scale_shape': (225, 300, 3), 'pad_shape': (225, 300, 3), 'scale_factor': 0.6, 'flip': False}, 
    {'ori_shape': (394, 500, 3), 'scale_shape': (236, 300, 3), 'pad_shape': (236, 300, 3), 'scale_factor': 0.6, 'flip': False}, 
    {'ori_shape': (392, 500, 3), 'scale_shape': (235, 300, 3), 'pad_shape': (235, 300, 3), 'scale_factor': 0.6, 'flip': False}, 
    {'ori_shape': (378, 500, 3), 'scale_shape': (227, 300, 3), 'pad_shape': (227, 300, 3), 'scale_factor': 0.6, 'flip': False}]
    """
    model = SSDVGG16()
    img = np.random.rand(4, 3, 236, 300)  # b,c,h,w
    img = torch.tensor(img).float()
    out3 = model(img)