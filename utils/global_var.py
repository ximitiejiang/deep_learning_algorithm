#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:51:59 2019

@author: ubuntu
"""

class GlobalVar():
    def __init__(self):
        """创建一个全局变量共享模块，可用于全局调试时共享调试标签，等其他需要全局变量的情况
        这样原始代码即使植入额外调试代码，也可以通过全局变量控制他在train/test时不执行
        这里借用的是gl这个对象来作为全局变量的存储载体, 所以是不是global_dict其实不重要，也可定义成self.global_dict
        两种设置方法：
        方法1. 直接在GlobalVar类的setting()里边设置
        方法2. 在对应的任何文件中先导入gl对象，然后调用gl.set_value()来进行设置
        而获得全局变量的方法就是gl.get_value(key_str)
        
        """
        global global_dict  
        global_dict = {}
        self.setting()
        
    def set_value(self, name, value):
        global_dict[name] = value
    
    def get_value(self, name, defValue=None):
        try:
            return global_dict[name]
        except KeyError:
            return defValue
    
    def setting(self):
        self.set_value("DEBUG_NMS_SCORE_THR", True)      # 可输出nms模块的对比，在bbox_nms.py
        self.set_value("DEBUG_NMS_NMS_THR", False)        # 可输出nms模块的对比，在bbox_nms.py
        self.set_value("DEBUG_ASSIGNER", False)           # 可输出assigner的读比，在

gl = GlobalVar()