#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:36:21 2019

@author: ubuntu
"""
import numpy as np

COLORS = dict(purple = [255, 0, 255],  # purple = magenta
              red = [255, 0, 0],
              green = [0, 255, 0],
              black = [0, 0, 0],
              cyan = [0, 255, 255],
              yellow = [255, 255, 0],
              blue = [0, 0, 255],
              white = [255, 255, 255])

def color2value(color_str):
    """定义一个把颜色字符串转换成pygame能识别的tuple
    注意：pygame认可的颜色顺序是rgb
    用法：color2value('green')
    """
    return COLORS[color_str]


# %% 分割任务调色板：颜色顺序就代表了标签号，比如第0行颜色就是第0个标签

def get_pallete(pallete):
    pallete_dict = {'voc': voc_pallete,
                    'city': city_pallete,
                    'ade': ade_pallete}
    return pallete_dict[pallete]
    
"""voc调色板：21种RGB(不是BGR)颜色对应21个类别0-20，其中0为背景，1-20为20个分类"""
voc_pallete = np.array([[  0,   0,   0],
                       [128,   0,   0],   # aerplane
                       [  0, 128,   0],   # bicycle
                       [128, 128,   0],   # bird
                       [  0,   0, 128],
                       [128,   0, 128],
                       [  0, 128, 128],
                       [128, 128, 128],
                       [ 64,   0,   0],
                       [192,   0,   0],
                       [ 64, 128,   0],
                       [192, 128,   0],
                       [ 64,   0, 128],
                       [192,   0, 128],
                       [ 64, 128, 128],
                       [192, 128, 128],
                       [  0,  64,   0],
                       [128,  64,   0],
                       [  0, 192,   0],
                       [128, 192,   0],
                       [  0,  64, 128]])


city_pallete = np.array([[128,  64, 128],
                       [244,  35, 232],
                       [ 70,  70,  70],
                       [102, 102, 156],
                       [190, 153, 153],
                       [153, 153, 153],
                       [250, 170,  30],
                       [220, 220,   0],
                       [107, 142,  35],
                       [152, 251, 152],
                       [  0, 130, 180],
                       [220,  20,  60],
                       [255,   0,   0],
                       [  0,   0, 142],
                       [  0,   0,  70],
                       [  0,  60, 100],
                       [  0,  80, 100],
                       [  0,   0, 230],
                       [119,  11,  32]])

ade_pallete = np.array([])
    
    
if __name__ == "__main__":
    pass
    