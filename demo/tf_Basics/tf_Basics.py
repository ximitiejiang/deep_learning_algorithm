#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:11:06 2019

@author: ubuntu
"""

"""
tensorflow教程(基于tensorflow2.0的教程)
教程参考：https://github.com/czy36mengfei/tensorflow2_tutorials_chinese (这个教程是主要以tf2.0为主的教程，比较完整)
教程参考：https://github.com/geektutu/tensorflow2-docs-zh (官方文档的私人中文版)

1. keras成为接口主力：包括datasets, layers, models都是从keras引入，在网络搭建上能够更加友好。
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


print(tf.__version__)             # 2.0.0
print(tf.keras.__version__)       # 2.2.4-tf

# %%
"""tf中的模型堆叠：
1. 采用keras的接口：tf.keras.Sequential()
"""
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


"""tf中每一层的设置：
1. 激活函数是作为层的参数放上去的：activation参数，默认不用任何激活函数
2. 初始化方法是作为层的参数放上去的：kernel_initializer/bias_initializer核初始化和偏置初始化，默认为Glorot uniform初始化方法
3. 正则化方法是作为层参数放上去的：kernel_regularizer/bias_regularizer核正则化和偏置正则化，默认不采用正则化函数
"""
layers.Dense(32, activation='sigmoid', kernel_initializer=tf.keras.initializers.glorot_normal)
layers.Dense(32, activation='sigmoid', kernel_regularizer=tf.keras.relularizers.l1(0.01))


"""tf中的训练：
1. 通过compile方法配置训练需要的模块
- optimizer
- loss
- metrics

"""
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy])

"""tf中正式训练

"""
train_x = np.random.randn((1000, 72))
train_y = np.random.randn((1000, 10))
model.fit(train_x, train_y, epochs=10, batch_size=100)


"""tf中进行结果评估
1. 采用
"""

