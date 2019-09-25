#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:01:12 2019

@author: ubuntu

永久加入路径的方法：
gedit ~/.bashrc
export PYTHONPATH="/home/ubuntu/suliang_git/deep_learning_algorithm/"

"""
import sys, os
path = os.path.abspath('.')
if not path in sys.path:
    sys.path.insert(0, path)


    