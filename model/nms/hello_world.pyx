#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:52:00 2019

@author: ubuntu

整个过程
1. 创建.pyx文件：里边包含相关调用
2. 创建setup.py文件：

"""

# 定义一个c语言函数
cdef extern from "stdio.h":
    extern int printf(const char *format, ...)

def sayHello():
    printf("hello, world!\n")