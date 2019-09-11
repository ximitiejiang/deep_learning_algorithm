#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:36:21 2019

@author: ubuntu
"""

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