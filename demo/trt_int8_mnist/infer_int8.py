#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:15:04 2019

@author: ubuntu
"""

"""                
采用int8数据来进行运算来提高计算速度，同时保证精度不下降很多。

### 量化基本概念
参考：https://zhuanlan.zhihu.com/p/58182172
1. 为什么要量化：模型太大，而weights范围固定且波动不大非常适合量化压缩，减少内存需求且减少计算量
2. 为什么量化有效：因为CNN对噪声不敏感???
3. 为什么不能直接用int8训练：因为学习率等东西很小，反向传播和梯度计算都涉及，无法用int8

4. 原理：为了把float32的卷积操作转换成int8的卷积操作，通常是通过校准，把float32的输出激活值映射到int8上。
   也就是准备一个校准数据集，用float32的模型去跑，统计每个层输出激活值的分布范围(float32)，然后把这个分布范围映射
   到int8的范围，注意不是float32的标称范围，而是实际算出来的float32的值的范围，量化公式就是
   fp32_tensor = scale_factor * int8_tensor + fp32_bias, 且英伟达说bias去掉对结果影响也不大。
   
   - 问题1：如果fp32分布不对称有什么问题？由于映射是基于取最大值且对称映射，也就是取一个最大值后同时在正负两边都用这个值来对称映射，
   这样能最大限度保留原信息。此时如果不对称，会导致int8区间也不对称，大部分范围没有对应值，浪费了int8本就不多的动态范围。
   - 问题2：如果fp32分布饱和有什么问题？此时也就是在设定阈值去避免分布不均匀问题之后，多出来的源数据造成的饱和问题，通常会定义一个阈值T,
   对称T来对fp32做映射，确保[-T, T]之间的fp32是基本对称的，并且使模型的精度损失最小。因此这里T的选取每层不同，且需要根据精度损失求解最优化问题
   - 问题3: 如何找这个设定阈值来让精度损失最小？这里采用KL散度(也就是相对熵)来评价不同T的损失情况，其中fp32作为初始最优编码，int8在不同T下作为优化后编码，
   对比不同编码下的损失情况。KL散度计算


### tensorRT的做法：是采用一个校准表来量化模型。
1. 如何生成校准表：需要输入500-1000张有代表性图片，最好每个类都要包括。
    - 把数据集转换成batch文件
    - 把batch文件输入tensorRT生成校准表

2. 如何使用校准表：校准表可以保存和读取，从而不用每次都生成一次校准表。采用writeCalibrationCache, readCalibrationCache

3. 采用int8 inference的原理
    - int8的精度和动态范围要小很多：int8(-128~+127), float16(-65504~65504), float32(-3.4x10^38~3.4x10^38), 所以从float32到int8需要不止一次的类型转换。
    - 采用的方式是内部用int8来计算，得到float32的结果
    - 计算方式：线性量化法，即fp_a = scale_a * qa
"""

import tensorrt as trt

class trt_cfg:
    model_path
    dataset_path
    DTYPE = trt.float32

class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """构建一个校准器：也就是构建一个校准数据集，校准原则是KL散度(相对熵)"""
    def __init__():
        # 初始化父类的构造函数： 注意虽然是继承父类，但父类
        super.__init__()
        # 如果事先已生成校准表则直接加载
        
        #
        self.data = load_mnist_data()
        self.batch_size = batch_size
        
    
    def read_calibration_cache(self):
        pass
    
    def write_calibration_cache(self, cache):
        pass
        
        

def inference_with_int8():
    """tensorRT定义了一个calibration校准器，使得在int8下的模型也能得到较好精度，从而int8模型获得更快计算速度(相比原来float32)"""
    # 加载数据集
    
    # 创建校准器(也就是校准数据集)
    calib = MNISTEntropyCalibrator)
    
    # 开始求取
    
    