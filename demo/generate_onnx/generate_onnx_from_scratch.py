#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:21:56 2019

@author: ubuntu
"""

"""
利用onnx自带的工具，从头创建一个onnx
参考：https://github.com/onnx/onnx/blob/f2daca5e9b9315a2034da61c662d2a7ac28a9488/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model
"""
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def build_onnx():
    # 创建输入 (ValueInfoProto)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    
    # 创建输出 (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    
    # 创建node,也就是创建一个层 (NodeProto)
    node_def = helper.make_node(
        'Pad', # node name
        ['X'], # inputs
        ['Y'], # outputs
        mode='constant', # attributes
        value=1.5,
        pads=[0, 1, 0, 1],
    )
    
    # 创建图，也就是一个模型参考 (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )
    # 创建模型 (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    
    print('The model is:\n{}'.format(model_def))
    onnx.checker.check_model(model_def)         # 官方的模型居然报错，跟我之前转yolov3一样，看来我之前注释这句是对的。
    print('The model is checked!')
    
    # 保存一个模型
    onnx.save(model_def, '/home/ubuntu/Desktop')


if __name__ =="__main__":
    build_onnx()