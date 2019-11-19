# 关于onnx
主要的参考资料如下：
```
onnx的官方python api: https://github.com/onnx/onnx/blob/f2daca5e9b9315a2034da61c662d2a7ac28a9488/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model
onnx的API中文介绍：https://blog.csdn.net/u013597931/article/details/84401047
onnx的proto结构介绍：https://oldpan.me/archives/talk-about-onnx
别人写的一个caffe2onnx转换器：http://www.pianshen.com/article/9295145984/
```

### onnx的protobuf定义
参考https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
onnx所定义的数据结构存放在一个onnx.proto的文件中，是基于google的protobuf协议，protobuf是采用类似于c++的语法格式来定义数据结构
如下是onnx.proto的一个简化结构
```
// copyright Facebook Inc. and Microsoft Corporation. 可见是facebook和微软一起，基于google的协议定制的数据结构
syntax = "proto2"；
// Nodes: 数据结构是NodeProto，也就是模型的每一层：conv, relu...
message NodeProto{
    repeated string input = 1;
    repeated string output = 2;
    optional string name = 3;
    optional string op_type = 4;
    repeated AttributeProto attribute = 5;
    optional string doc_string = 6;
    optional string domain = 7;
}
// Models: 数据结构是ModelProto，也就是整个模型最大的集合，包含图和版本信息等
message ModelProto{
    optional int64 ir_version = 1;
    optional string producer_name = 2;
    optional string producer_version = 3;
    optional string domain = 4;
    optional int64 model_version = 5;
    optional string doc_string = 6;
    optional GrapProto graph = 7;
}

// Graphs: 数据结构是GraphProto，也就是模型的构造和权重(最核心部分)
message GraphProto{
    repeated NodeProto node = 1;
    optional string name = 2;
    repeated TesnorProto initializer = 5;  // 即所有权重
    ...
}
```


### 创建一个最简单的onnx模型
创建onnx模型分三步：把冰箱门打开，把onnx拿出来，把冰箱门关上...哦不是，是先创建node，再创建graph，最后创建model
通常采用onnx的python api进行onnx模型的创建，可以参考tensorRT中yolov3的onnx模型创建
```
// 这里简单onnx模型创建参考自onnx官网tu
```

### 已生成的onnx模型的认识

```
ir_version: 1
producer_name: "pytorch"
producer_version: "0.2"
domain: "com.facebook"
// graph信息：包含每个node
graph {
  // 第一个node：也就是网络第一层，为一个conv  
  node {
    input: "1"
    input: "2"
    output: "11"
    op_type: "Conv"
    attribute {
      name: "kernel_shape"
      ints: 5
      ints: 5
    }
    attribute {
      name: "strides"
      ints: 1
      ints: 1
    }
    attribute {
      name: "pads"
      ints: 2
      ints: 2
      ints: 2
      ints: 2
    }
    attribute {
      name: "dilations"
      ints: 1
      ints: 1
    }
    attribute {
      name: "group"
      i: 1
    }
  }
  // 第二个node: 也就是网络第二层, 为一个add操作
  node {
    input: "11"
    input: "3"
    output: "12"
    op_type: "Add"
    attribute {
      name: "broadcast"
      i: 1
    }
    attribute {
      name: "axis"
      i: 1
    }
  }
  // 第三个node：也就是第三层，为一个relu
  node {
    input: "12"
    output: "13"
    op_type: "Relu"
  }
}
  ...
// graph之外的其他信息
name: "torch-jit-export"
initializer {
dims: 64
dims: 1
dims: 5
dims: 5
data_type: FLOAT
name: "2"
raw_data: "\034
```