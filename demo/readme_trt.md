# 用tensorRT进行深度神经网络的部署


### 关于tensorRT
1. 当前最新版本是tensorRT6.0, 需要匹配cuda9.0+cudnn7.6.3以上, 或者cuda9.2+cudnn7.6.3以上。
由于我安装的cuda9.0是conda里边的配置是cuda9.0+cudnn7.1.4，他没有新版本的cuda9.0+cudnn7.6.3，所以cudnn也就没法升级除非把cuda版本提上去，但担心有其他不匹配问题，所以暂时不升了。


### C++ API和python API
1. 一般来说，c++和python api都很类似都能支持项目需求，但在一些性能要求很高(实时性敏感)，安全要求很高(自动驾驶)的场景，c++是更好选择。
2. c++ api是可以使用在所有平台的(linux x86-64, windows, linux ppc64le, linux AArch64, QNX)，但python api不能在windows和QNX平台使用

### 创建trt logger
logger = trt.Logger(trt.Logger.WARNING)


### 创建trt的network模型
1. 方式1：从头创建一个trt network，此时需要通过network API的add_convolution()等方法创建。
2. 方式2：从caffe/tensorflow/onnx导入和解析
    - onnx发展迅速，有可能产生onnx模型版本跟parser解析器版本不匹配的问题 ——是否需要自己写一个onnx转换器？
    - tensorRT的onnx后端，参考：https://github.com/onnx/onnx-tensorrt/blob/master/README.md
    - 导入的过程主要就是创建3个对象：构造器，网络模型，解析器
        - 创建builder: trt.Builder(logger) as builder
        - 创建network: builder.create_network() as network
        - 创建parser: trt.OnnxParser(network, logger) as parser


### 创建trt的engine对象
1. engine对象是把network的结构和权重和计算进行优化之后的结果模型。
2. 采用builder对象来创建engine, builder对象需要先设置好max_batch_size, max_workspace_size, 生成config, 然后就可以生成engine.
3. 创建好engine之后，权重就会拷贝到engine，之前的network等数据就可以销毁了。

```
builder.max_batch_size = max_batch_size
builder.max_workspace_size = 1 << 20 # 该值决定了创建一个优化的engine时的可用内存，设置越高越好
with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.build_cuda_engine(network, config) as engine:
```


### 模型序列化与反序列化
1. 序列化一个模型是指把engine转化为一个格式化可存储的数据结构，方便后续进行inference，从而避免每次都要把模型转化为engine(比较费时)，
而是先保存序列化模型，然后反序列化模型为engine则很快。最终的inference则是用engine来进行。
2. 序列化的模型并不具有通用性，他跟操作体统，GPU类型，tensorRT版本都有关系。
3. 序列化模型生成：就是生成modelstream, 这种序列化模型也叫作plan file, 所以生成序列化模型的子程序也叫作make_plan()
```
seri_model = engine.serialize()
```

4. 反序列化模型：也就是重新把序列化模型转换成engine用来进行推理。
    - 反序列化需要先创建一个runtime运行时对象


5. 保存序列化模型和读取序列化模型：
```
with open(“seri_data.engine”, “wb”) as f:  # 打开文件
    f.write(engine.serialize())     # 写入文件
    
with open(“seri_data.engine”, “rb”) as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
```

### 模型的推理inference
1. 首先需要为输入输出创建缓存
```
h_input = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.float32) 　　#　为主机的输入输出创建锁页缓存(不会跟disk进行数据交换) 
h_output = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.float32)

d_input = cuda.mem_alloc(h_input.nbytes)  # 给输入输出分配设备内存
d_output = cuda.mem_alloc(h_output.nbytes)

stream = cuda.Stream()  #创建一个流来拷贝输入输出并进行推理

with engine.create_execution_context() as context:  #进一步创建一个context对象，存放计算过程生成的激活值
	cuda.memcpy_htod_async(d_input, h_input, stream)   #把输入数据送如gpu
	context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)  # 在context对象中进行推理计算
	cuda.memcpy_dtoh_async(h_output, d_output, stream)  # 把计算结果返回给gpu
	stream.synchronize()    # 同步流中的输入输出
    return h_output　　　　  # 返回输出
```



### 调试：
1. 报错：pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?
也就是在预分配host，device内存的时候就报错了h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(cfg.DTYPE))

问题原因：pycuda.driver没有初始化，导致无法得到context，需要在导入pycuda.driver后再导入pycuda.autoinit
```
import pycuda.driver as cuda
import pycuda.autoinit
```

2. 报错：UnicodeDecodeError: 'utf-8' codec can't decode byte 0xaa in position 8: invalid start byte
也就是在逆序列化已保存的engine加载到内存中时，在engine = runtime.deserialize_cuda_engine(f.read())时报错

问题原因：在打开导入序列化模型时，需要采用'rb'模式才能读，否则不能读取，即在读取序列化模型时，需要做3件事
    - 打开文件，必须用rb模式：with open(cfg.work_dir + 'serialized.engine', 'rb') as f
    - 创建runtime：trt.Runtime(logger) as runtime
    - 基于runtime生成反序列化模型：engine = runtime.deserialize_cuda_engine(f.read())




### 关于onnx
1. onnx后缀的文件，内核是基于protobuf进行数据结构定义的，通过了解protobuf的知识，可以知道proto文件的编译和读写方式。
2. onnx文件，都共享一个onnx.proto文件，这就是基于protobuf生成的模型数据结构，该proto文件基本内容包括：
```
// copyright Facebook Inc. and Microsoft Corporation. 可见是facebook和微软一起，基于google的协议定制的数据结构
syntax = "proto2"；
// Nodes: 也就是模型的每一层：conv, relu...
message NodeProto{
    repeated string input = 1;
    repeated string output = 2;
    optional string name = 3;
    optional string op_type = 4;
    repeated AttributeProto attribute = 5;
    optional string doc_string = 6;
    optional string domain = 7;
}
// Models: 也就是整个模型最大的集合，包含图和版本信息等
message ModelProto{
    optional int64 ir_version = 1;
    optional string producer_name = 2;
    optional string producer_version = 3;
    optional string domain = 4;
    optional int64 model_version = 5;
    optional string doc_string = 6;
    optional GrapProto graph = 7;
}

// Graphs: 也就是模型的构造和权重(最核心部分)
message GraphProto{
    repeated NodeProto node = 1;
    optional string name = 2;
    repeated TesnorProto initializer = 5;  // 即所有权重
    ...
}
```

3. 读取一个onnx文件：如下是一个已经读取出来的onnx模型如下，可见他的数据是安装onnx.proto定义的结构进行存放的：
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

### 如何用python来优化性能
参考：nvidia的tensorRT developer guide手册中的'How do i optimize my python performance?'
