### 关于tensorRT


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
1. 序列化一个模型是指把engine转化为一个格式化可存储的数据结构，方便后续进行inference，从而避免每次都要把模型转化为engine(比较费时)，而是先
保存序列化模型，然后反序列化模型为engine则很快。最终的inference则是用engine来进行。
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

### 模型的部署
1. 模型的部署，是指对序列化模型也就是plan，拷贝到目标机器上，执行相关的分类、回归、检测、分割任务。前面都是make_plan()，后边就是execute_plan()
2. 


### 如何用python来优化性能
参考：nvidia的tensorRT developer guide手册中的'How do i optimize my python performance?'
