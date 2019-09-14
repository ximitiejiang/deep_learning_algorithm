#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:35:56 2019

@author: ubuntu
"""
from terminaltables import AsciiTable

def test_AsciiTable():
    """用AsciiTable可以制作最简单的字符表格：如下3步即可生成输出。
        - table_data = [[], [], [],...]  # 创建数据
        - table = AsciiTable(table_data) # 生成表格
        - print(table.table)             # 打印表格
    """
    header = ['key', 'expected shape', 'loaded shape']
    table_data = [header] + [['aa', (3,20,20), (3,40,40)]]
    table =AsciiTable(table_data)
    
    print('these shape mismatched:')
    print(table.table)

if __name__ == '__main__':
    test_AsciiTable()