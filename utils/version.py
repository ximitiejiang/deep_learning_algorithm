#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:34:43 2019

@author: ubuntu
"""
import platform
import cv2
import numpy as np
import torch
import torchvision

def check_version():
    print("Python version: ", platform.python_version())
    print("opencv version: ", cv2.__version__)
    print("numpy version: ", np.__version__)
    print("pytorch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)    

if __name__ == "__main__":
    check_version()