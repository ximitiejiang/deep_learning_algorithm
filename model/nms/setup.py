#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:53:41 2019

@author: ubuntu
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

setup(
      name = "Hello world app",
      ext_modules = cythonize([
              Extension("hello",["hello_world.pyx"]),
              ]),
)