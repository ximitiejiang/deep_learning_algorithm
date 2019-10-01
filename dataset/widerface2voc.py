#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:30:36 2019

@author: ubuntu
"""

"""
wider face原始的图片存放方式如下：
0--Parads
    img1, img2...
1--xxx
    img1, img2...
但参考mmdetection得到的annotation文件中的路径则只包含文件名，不包含外边那层文件夹壳子。
解决方式1：基于文件名生成文件夹名加到路径中去打开文件，但发现文件夹名在文件名中形式不统一，很难提取。

解决方式2：预先准备所有文件夹名称，然后直接把这些文件夹加到文件名中去(当前采用的方式)

解决方式3：把文件夹中的文件都拷贝到同一个文件夹中，直接调用。如下就是这种方式的实现：
"""

import os
import shutil

#目标文件夹，此处为相对路径，也可以改为绝对路径
destination = 'WIDER_train/imgs/'
if not os.path.exists(destination):
    os.makedirs(destination)

#源文件夹路径
path = 'WIDER_train/images/'
folders= os.listdir(path)
for folder in folders:
    dir = path + '/' +  str(folder)
    files = os.listdir(dir)
    for file in files:
        source = dir + '/' + str(file)
        deter = destination + str(file)
        shutil.copyfile(source, deter)
