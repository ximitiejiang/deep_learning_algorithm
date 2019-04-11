#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:16:08 2019

@author: ubuntu
"""

import cv2

__all__ = ['color2value', 'bgr2rgb', 'rgb2bgr', 'bgr2hsv', 'hsv2bgr','bgr2gray', 'gray2bgr']

colors = dict(purple = [255, 0, 255],  # purple = magenta
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
    return colors[color_str]
    

def convert_color_factory(src, dst):
    """
    Args:
        src(str): The input image color type.
        dst(str): The output image color type.
    Returns:
        convert_color(ndarray): The converted image
    """
    code = getattr(cv2, 'COLOR_{}2{}'.format(src.upper(), dst.upper()))
    # 这部分COLOR开头的变量作为dict的key存储在cv2中,也可直接写cv2.COLOR_BGR2RGB

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    return convert_color


def bgr2gray(img, keepdim=False):
    """Convert a BGR image to grayscale image.

    Args:
        img (ndarray): The input image.
        keepdim (bool): If False (by default), then return the grayscale image
            with 2 dims, otherwise 3 dims.

    Returns:
        ndarray: The converted grayscale image.
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keepdim:  # 默认灰度图输出是二维(w, h)
        out_img = out_img[..., None]
    return out_img


def gray2bgr(img):
    """Convert a grayscale image to BGR image.

    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted BGR image.
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return out_img

bgr2rgb = convert_color_factory('bgr', 'rgb')

rgb2bgr = convert_color_factory('rgb', 'bgr')

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    path = '../repo/test.jpg'
    img = cv2.imread(path) # (h,w,c)-bgr
    img1 = bgr2hsv(img)
    img2 = bgr2rgb(img)
    img3 = bgr2gray(img)
    
    plt.subplot(151)
    plt.title('bgr')
    plt.imshow(img)
    
    plt.subplot(152)
    plt.title('rgb')
    plt.imshow(img2)
    
    plt.subplot(153)
    plt.title('hsv')
    plt.imshow(img1)
    
    plt.subplot(154)
    plt.title('gray')
    plt.imshow(img3)
    