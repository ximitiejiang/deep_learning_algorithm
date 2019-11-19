#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:12:18 2019

@author: ubuntu
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import torch
import torchvision
#import onnxmltools
#from onnxmltools.utils.float15_converter import convert_float_to_float16
    
def create_onnx_components():
    """创建onnx的基本组件：node, graph, model，可以看到onnx是如何本onnx.proto文件对应的
    参考：https://github.com/onnx/onnx/blob/master/onnx/examples/Protobufs.ipynb
    """
    # ------ 创建int变量: 传入数值和描述即可 ------ 
    arg1 = helper.make_attribute("this is INT", 64)
    
    arg2 = helper.make_attribute("this is float/1", 3.14)
    
    arg3 = helper.make_attribute("this is STRING", "helloworld")
    
    arg4 = helper.make_attribute("this is INTS", [1,2,3,4])
    
    # ------ 创建TensorProto ------
    tensor0 = helper.make_tensor_value_info()   # ？
    
    array1 = np.array([[1,2,3],[4,5,6]])
    tensor1 = numpy_helper.from_array(array1)   # 从numpy获取tensorProto
    
    with open('ts.pb', 'wb') as f:
        f.write(tensor1.SerializeToString())     # 保存tensorProto
    tensor2 = TensorProto()
    with open('ts.pb', 'rb') as f:
        tensor2.ParseFromString(f.read())       # 读取tensorProto
    with
    
    
    # ------ 创建node ------ 
    node1 = helper.make_node("Relu", ["X"], ["Y"])  # op_type="Relu"
    
    node2 = helper.make_node("Conv", ["X", "W", "Y"], kernel=3, stride=1, pad=1)
    print(node2)
    print(helper.printable_node(node2))    #　这就是常看到的onnx形式：%Y = Conv[]
    
    # ------ 创建graph ------ 
    node_list = []
    arg_list = []
    graph1 = helper.make_graph(
    [
        helper.make_node("FC", ["X", "W1", "B1"], ["H1"]),
        helper.make_node("Relu", ["H1"], ["R1"]),
        helper.make_node("FC", ["R1", "W2", "B2"], ["Y"]),
    ],
    "MLP",
    [
        helper.make_tensor_value_info('X' , TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('W1', TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('B1', TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('W2', TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('B2', TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1]),
    ])

    
def create_onnx_model():
    """创建一个完整的onnx模型
    参考：https://zhuanlan.zhihu.com/p/41255090
    """   
    # 创建输入 (ValueInfoProto)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    # 创建输出 (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    # 创建node (NodeProto)
    node_def = helper.make_node(
        'Pad', # node name
        ['X'], # inputs
        ['Y'], # outputs
        mode='constant', # attributes
        value=1.5,
        pads=[0, 1, 0, 1],
    )
    # 创建图 (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )
    # 创建模型 (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    
    print('The model is:\n{}'.format(model_def))
    onnx.checker.check_model(model_def)
    print('The model is checked!')


def transfer_from_pytorch_to_onnx():
    """把pytorch模型转换到onnx
    参考：https://pytorch.org/docs/master/onnx.html
    """
    dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
    model = torchvision.models.alexnet(pretrained=True).cuda()
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
    
    # 然后也可以用onnx打开和检查和打印
    onnx_model = onnx.load(alexnet.onnx)
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)


def onnx_check():
    """对生成的onnx进行检查
    参考：https://github.com/onnx/tutorials/blob/master/tutorials/CorrectnessVerificationAndPerformanceComparison.ipynb
    """
    onnx_model = onnx.load("/home/ubuntu/mytrain/onnx_yolov3/yolov3_coco.onnx")
    print(helper.printable_graph(onnx_model.graph))  # 把onnx先打印出来看看
    onnx.checker.check_model(onnx_model)    # 报错onnx.onnx_cpp2py_export.checker.ValidationError: Node (086_upsample) has input size 2 not in range [min=3, max=4].


def onnx_version_conversion():
    """onnx的版本转换，采用onnx的verson converter
    参考：https://github.com/onnx/tutorials/blob/master/tutorials/VersionConversion.md
    """
    model = onnx.load("old_model_v9.onnx")
    onnx.checker.check_model(model)  # 检查IR已经很好的形成
    from onnx import version_converter
    converted_model = version_converter.convert_version(model, 8)  # 这里把onnx模型从version9转到version8
    onnx.save(converted_model, "new_model_v8.onnx")
    
    

def float32_to_float16():
    """用于把float32模型转化为float16模型：在模型量化过程中采用float16可以很大提高速度
    onnx自带了一个工具可以进行转换：
    import onnxmltools
    from onnxmltools.utils.float15_converter import convert_float_to_float16
    
    参考：https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/float32_float16_onnx.ipynb
    """
    # 定义输入和输出模型名称
    input_onnx_model = 'model.onnx'
    output_onnx_model = 'model_f16.onnx'
    # 加载onnx模型
    onnx_model = onnxmltools.utils.load_model(input_onnx_model)
    # 转换
    onnx_model = convert_float_to_float16(onnx_model)
    # 保存
    onnxmltools.utils.save_model(onnx_model, output_onnx_model)
    
    
if __name__ == "__main__":
    onnx_check()