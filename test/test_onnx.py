#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:00:12 2019

@author: ubuntu
"""
import torch
import torchvision
import numpy as np
import os
import glob
import torch as backend


"""
这部分用于对onnx模型进行验证：采用onnx模型自带的input, output数据
1. 安装onnx
需要预安装onnx: (参考https://github.com/onnx/onnx)，conda install -c conda-forge onnx
通过conda就会自动安装libprotobuf和protobuf和onnx
其中安装protobuf之后，就会生成3类静态和动态库文件：libprotobuf, libprotobuf-lite, libprotoc

问题1： 运行import onnx报错ImportError: libprotobuf.so.20: cannot open shared object file: No such file or directory
似乎是我conda帮我安装的protobuf的版本不对，所以生成的libprotobuf.so.21(在/home/suliang/anaconda3/lib/中), 而不是libprotobuf.so.20.
需要把protobuf降一个版本。于是采用conda的图形界面anaconda-navigator，筛选protobuf出来2个库，一个是libprotobuf, 另一个protobuf,
同时选择把他们降级从3.10.0到3.9.2，然后conda就会自动安装相应的库，安装完后检查anaconda3/lib，确实从libprotobuf.so.21变成了libprotobuf.so.2o。
最后问题消失import onnx正常了。
    - protobuf3.6 -> libprotobuf.so.17
    - protobuf3.7 -> libprotobuf.so.18
    - 查看protobuf版本: protoc --version
    - 查看默认调用的protobuf: which protoc
    - 查看哪些路径安装了protoc: whereis protoc
    - 查看系统安装的所有的protobuf: locate libprotobuf.so, 只有这条命令我才发现我机器安装了libprotobuf.so, libprotobuf.so.9, libprotobuf.so.9.0.1
        然后在anaconda3/lib路径下还有libprotobuf.so.21，就是没有哪个libprotobuf.so.20，直到自己手动降级protobuf的版本才得到libprotobuf.so.20。

2. protobuf的特点：protobuf是google开发的高效存储读取结构化数据工具，比xml,json更高效，且把数据压缩得更小(json的1/10，xml的1/20)
    - 


3. onnx模型的运行
    - 可以用onnx自己的onnx runtime，需要安装onnx runtime
        import onnxruntime as ort
        ort.run()
    - 可以用其他框架自己的runtime: 比如MNN的，caffe的，pytorch的(?)
        
"""
if __name__ =='__main__':
    
    task = 'validate'
    
    # 导出模型为onnx(只需要安装pytorch)
    if task == 'export':
        # 定义待保存的模型
        model = torchvision.models.alexnet(pretrained=True).cuda()
        # 定义待保存的test data
        input = torch.randn(10, 3, 224, 224, device='cuda')
        input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
        output_names = [ "output1" ]
        torch.onnx.export(model, input, "alexnet.onnx", verbose=True, 
                          input_names=input_names, output_names=output_names)  # 导出的onnx模型一般都很小，5M一下
    
    # 用onnx原生算法验证模型的输入输出(需要安装onnx)
    if task == 'validate':
        import onnx
        from onnx import numpy_helper
        
        model_dir = '/home/ubuntu/MyWeights/onnx/tiny_yolov2/Model.onnx'
        test_data_dir = '/home/ubuntu/MyWeights/onnx/tiny_yolov2/test_data_set_0'
        # 加载模型
        model = onnx.load(model_dir)
        # 加载输入
        inputs = []
        inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
        for i in range(inputs_num):
            input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
            tensor = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            inputs.append(numpy_helper.to_array(tensor))
        
        # 加载输出
        ref_outputs = []
        ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
        for i in range(ref_outputs_num):
            output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
            tensor = onnx.TensorProto()
            with open(output_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            ref_outputs.append(numpy_helper.to_array(tensor))
        
        # Run the model on the backend
        outputs = list(model(input) for input in inputs)
        
        # Compare the results with reference outputs.
        for ref_o, o in zip(ref_outputs, outputs):
            np.testing.assert_almost_equal(ref_o, o)
            
            
            