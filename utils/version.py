#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 08:27:52 2019

@author: ubuntu
"""

import platform
import cv2
import numpy as np
import torch
import torchvision
import tensorflow
import caffe

"""
python: 3.7.3
opencv: 4.1.1
numpy: 1.16.2
pytorch: 1.1.0
torchvision: 0.3.0
tensorflow: 2.0.0
caffe: 1.0.0
"""

def get_versions():
    versions = dict(
            python = platform.python_version(),
            opencv = cv2.__version__,
            numpy = np.__version__,
            pytorch = torch.__version__,
            torchvision = torchvision.__version__,
            tensorflow = tensorflow.__version__,
            caffe = caffe.__version__)
    for name, version in versions.items():
        print('{}: {}'.format(name, version))
    return versions


if __name__ == "__main__":
    get_versions()