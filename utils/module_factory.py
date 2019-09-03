#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:28:04 2019

@author: ubuntu
"""


# %% 注册类
class Registry():  # registry表示注册处
    
    def __init__(self):
        self.module_dict = dict()
    
    def register_module(self, cls):  # 注意：该注册函数生效的前提是类先要在__init__中导入，相当于初次运行
        """作为装饰器接收类"""
        module_name = cls.__name__
        self.module_dict[module_name] = cls
        return cls

registry = Registry()  # 实例化注册处，里边就会存放已注册的类



# %% 实例化对象
def build_module(cfg, registry):
    class_name = cfg.pop("type")   # 提取类的名称字符串
    if class_name not in registry.module_dict:
        raise KeyError('{} is not in the registered module dict'.format(class_name))
    else:
        class_obj = registry.module_dict[class_name]   # 获得类
    return class_obj(**cfg)                            # 返回实例化的类对象





if __name__ == "__main__":
    from addict import Dict
    cfg = Dict(type = "TestClass")
    obj = build_module(cfg, registry)