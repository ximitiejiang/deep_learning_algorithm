#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:48:14 2019

@author: ubuntu
"""
import numpy as np
import torch

# %%
class AnchorGenerator():
    """生成base anchors和grid anchors
    其中base anchor的生成方式，可以通过基础参数计算得到，也可以直接提供base anchor(比如线下通过kmean获得的base anchor)
    """
    def __init__(self, base_size=None, scales=None, ratios=None, ctr=None, 
                 scale_major=False, base_anchors=None):
        self.base_size = base_size
        self.scales = np.array(scales)
        self.ratios = np.array(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        
        # 计算base_anchors
        if base_anchors is None:
            self.base_anchors = self.get_base_anchors()  # 采用常规的方式生成base anchors
        else:
            self.base_anchors = base_anchors   # 提供通过kmean生成的base anchors
    
    def get_base_anchors(self): 
        """生成单个特征图的base anchors
        """
        w, h = self.base_size, self.base_size
        # 准备中心点坐标
        if self.ctr is None:
            x_ctr, y_ctr = 0.5 * (w - 1), 0.5 * (h - 1) 
        else:
            x_ctr, y_ctr = self.ctr
        # 准备宽高的比例: ratio=r则h=sqrt(r)*base_size, w=1/sqrt(r) * base_size，从而h/w=r
        h_ratios = np.sqrt(self.ratios) #(n,)
        w_ratios = 1 / h_ratios         #(n,)
        # 计算变换后的w', h'
        if self.scale_major:
            w_new = (w * w_ratios[:, None] *self.scales[None, :]).reshape(-1)    # (n,1)*(1,m)->(n,m)->(n*m,)
            h_new = (h * h_ratios[:, None] *self.scales[None, :]).reshape(-1)
        else:
            w_new = (w * self.scales[:, None] * w_ratios[None, :]).reshape(-1)    # (m,1)*(1,n)->(m,n)->(m*n,)
            h_new = (h * self.scales[:, None] * h_ratios[None, :]).reshape(-1)
        # 计算坐标xmin,ymin,xmax,ymax
        base_anchors = np.stack([x_ctr - 0.5 * (w_new - 1), 
                                 y_ctr - 0.5 * (h_new - 1),
                                 x_ctr + 0.5 * (w_new - 1), 
                                 y_ctr + 0.5 * (h_new - 1)], axis=-1).round()  # (m*n, 4))
        
        return torch.tensor(base_anchors, dtype=torch.float32)  # 做类型转换为pytorch weight通用的float32(否则会因为Numpy原因变float64)
    
    
    def grid_anchors(self, featmap_size, stride, device=torch.device('cuda')):
        """生成单个特征图的网格anchors
        """
        #确保生成的anchor跟特征结果在同一device
        base_anchors = self.base_anchors.to(device) #(k, 4)
        # 生成原图上的网格坐标
        h, w = featmap_size
        x = np.arange(0, w) * stride  # (m,)
        y = np.arange(0, h) * stride  # (n,)
        xx = np.tile(x, (len(y),1)).reshape(-1)                # (m*n,)
        yy = np.tile(y.reshape(-1,1), (1, len(x))).reshape(-1) # (m*n,)
        # 基于网格坐标生成(xmin, ymin, xmax, ymax)的坐标平移矩阵
        shifts = np.stack([xx, yy, xx, yy], axis=-1)  # (m*n, 4)
        shifts = torch.tensor(shifts).type_as(base_anchors).to(device)
        # 平移anchors: 相当于该组anchors跟每个平移坐标进行相加，
        # 也就相当于要取每一行坐标跟一组anchor运算，所以用坐标插入空轴而不是用anchor插入空轴
        all_anchors = base_anchors + shifts[:, None, :]   #(b,4)+(k,1,4)->(k, b, 4)
        all_anchors = all_anchors.reshape(-1, 4)  # (k*b, 4)
        return all_anchors


if __name__ == "__main__":
    """base_anchor的标准数据
    [[-11., -11.,  18.,  18.],[-17., -17.,  24.,  24.],[-17.,  -7.,  24.,  14.],[ -7., -17.,  14.,  24.]]
    [[-22., -22.,  37.,  37.],[-33., -33.,  48.,  48.],[-34., -13.,  49.,  28.],[-13., -34.,  28.,  49.],[-44.,  -9.,  59.,  24.],[ -9., -44.,  24.,  59.]]
    [[-40., -40.,  70.,  70.],[-51., -51.,  82.,  82.],[-62., -23.,  93.,  54.],[-23., -62.,  54.,  93.],[-80., -16., 111.,  47.],[-16., -80.,  47., 111.]]
    [[ -49.,  -49.,  112.,  112.],[ -61.,  -61.,  124.,  124.],[ -83.,  -25.,  146.,   88.],[ -25.,  -83.,   88.,  146.],[-108.,  -15.,  171.,   78.],[ -15., -108.,   78.,  171.]]
    [[ -56.,  -56.,  156.,  156.],[ -69.,  -69.,  168.,  168.],[-101.,  -25.,  200.,  124.],[ -25., -101.,  124.,  200.]]
    [[ 18.,  18., 281., 281.],[  6.,   6., 293., 293.],[-37.,  57., 336., 242.],[ 57., -37., 242., 336.]]
    """
    
#    import sys, os
#    path = os.path.abspath("../utils")
#    if not path in sys.path:
#        sys.path.insert(0, path)
    from utils.visualization import vis_bbox
    name = 'test1'
    
    if name == 'test1':  # 检测生成的base anchor是否正确
        base_sizes = [30,60,112,165,217,270]
        anchor_scales = [(1, 1.4)]
        anchor_ratios = [(1,1/2,2),(1,1/2,2,1/3,3)]
        centers = [(3.5, 3.5)]
        anchor_generators = []
        for i in range(len(base_sizes)):
            anchor_generator = AnchorGenerator(base_sizes[i],
                                               anchor_scales[i],
                                               anchor_ratios[i],
                                               ctr=centers[i],
                                               scale_major=False)
            base_anchors = anchor_generator.base_anchors
            vis_bbox(base_anchors)
            anchor_generator.base_anchors = anchor_generator
            anchor_generators.append(anchor_generator)
            