# 用tensorRT进行深度神经网络的部署


### 关于tensorRT
1. 当前最新版本是tensorRT6.0, 需要匹配cuda9.0+cudnn7.6.3以上, 或者cuda9.2+cudnn7.6.3以上。
安装主要参考官方文档，采用tar安装包安装的。中文版安装指南可以参考oldpan博客：https://oldpan.me/archives/use-tensorrt-speed-up-deep-learning-1
由于我安装的cuda9.0是conda里边的配置是cuda9.0+cudnn7.1.4，他没有新版本的cuda9.0+cudnn7.6.3，所以cudnn也就没法升级除非把cuda版本提上去，但担心有其他不匹配问题，所以暂时不升了。
tensorRT的安装比较简单，只要之前通过conda安装过cuda toolkit，则下载tar安装包(deb形式容易安装失败)，解压缩，并把路径导入系统路径文件中，这样c++ api就可以使用了。
如果需要python api，则再进入解压缩包里边的python文件夹，运行其中一个跟自己系统匹配的whl文件，我是采用：
`pip install tensorrt-6.0.1.5-cp37-none-linux_x86_64.whl`
然后安装Pythongraphsugeon:
```
cd TensorRT-6.x.x.x/graphsurgeon
pip install graphsurgeon-0.4.1-py2.py3-none-any.whl
```
然后安装pycuda: `pip install 'pycuda>=2019.1.1'`
最后检查是否安装成功：`python    import tensorrt as trt`

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
        - 创建builder: trt.Builder(logger) as builder，该builder对象用来操作network生成engine
        - 创建network: builder.create_network() as network, 该network是按照
        - 创建parser: trt.OnnxParser(network, logger) as parser，
    - 生成过程
        - parser.parse(model.read())   这个过程就是先把模型读取出来，然后解析的过程就是把模型权重填充进network
        - engine = builder.build_cuda_engine(network)   这个过程就是把network再打包成engine, 但在过程中会进行一系列优化操作，比如对conv的计算过程进行优化
        
        

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

### 关于读取onnx模型
1. onnx模型的可视化：可采用netron


### 如何用python来优化性能
参考：nvidia的tensorRT developer guide手册中的'How do i optimize my python performance?'



### 关于模型量化
1. 什么是模型量化：


### 调试：
1. 报错：pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?
也就是在预分配host，device内存的时候就报错了h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(cfg.DTYPE))

问题原因：pycuda.driver没有初始化，导致无法得到context，需要在导入pycuda.driver后再导入pycuda.autoinit
```
import pycuda.driver as cuda
import pycuda.autoinit
```

2. 报错：UnicodeDecodeError: 'utf-8' codec can't decode byte 0xaa in position 8: invalid start byte
也就是在逆序列化已保存的engine加载到内存中时，在engine = runtime.deserialize_cuda_engine(f.read())时报错。
问题原因：在打开导入序列化模型时，需要采用'rb'模式才能读，否则不能读取，即在读取序列化模型时，需要做3件事
    - 打开文件，必须用rb模式：with open(cfg.work_dir + 'serialized.engine', 'rb') as f
    - 创建runtime：trt.Runtime(logger) as runtime
    - 基于runtime生成反序列化模型：engine = runtime.deserialize_cuda_engine(f.read())


3. 报错：yolov2_onnx模型出来的预测结果总是置信度太低，没有准确性可言。
问题原因：
    - 自己的代码里边对输出的bbox没有做处理就送去显示了，但实际上yolov2的输出形式是(xmin,ymin,w,h)，需要转化为(xmin,ymin,xmax,ymax)
    - 自己的代码里边多写了对预测图片的归一化操作，yolov2的实际的预测过程不需要归一化（但其他的是需要的，比如yolov3）。


4. 报错：onnx.onnx_cpp2py_export.checker.ValidationError: Op registered for Upsample is deprecated in domain_version of 11.
==> Context: Bad node spec: input: "085_convolutional_lrelu" output: "086_upsample" name: "086_upsample" op_type: "Upsample" attribute 
{ name: "mode" s: "nearest" type: STRING } attribute { name: "scales" floats: 1 floats: 1 floats: 2 floats: 2 type: FLOATS }
问题原因：onnx更新太快了，在官方1.5.1以后就取消了upsample层，所以对yolov3报错了。而我的onnx是安装的1.6.1，不过话说回来upsample取消，那用啥? interpolate? 那就要么更改yolov3的模型换掉upsample然后重新训练，要么没辙。
参考https://devtalk.nvidia.com/default/topic/1052153/jetson-nano/tensorrt-backend-for-onnx-on-jetson-nano/1
修改方式是先降级onnx到1.4.1
```
pip uninstall onnx
pip install onnx==1.4.1
```
可惜我用conda安装的onnx，居然没有旧版本可用，想conda uninstall onnx，提示我要下载600M的东西，我x，安装时10M不到，卸载要动600M...


5. 报错：[TensorRT] ERROR: Network must have at least one output。该步报错发生在engine = builder.build_cuda_engine(network)
问题原因：
- 可能原因1：没有output，则手动标记output, 首先查看相关变量
    parser.parse(model.read())   　　　　　　　 #首先解析模型，此时就会把权重填充到network，也就能从network查看解析结果了。(如果在解析前查看network就是一个空壳)
    network.num_layers                         #可查看模型有多少层: 比如显示61则说明有61层(0-60)，发现我的模型只解析到60层，也就是backbone层，后边的层都没有解析出来
    network.get_layer(60).get_output(0).shape  #可看到最后一层的输出

    network.mark_output(network.get_layer(network.num_layers-1).get_output(0))   #手动标记输出层


6. 报错：[TensorRT] ERROR: Network must have at least one output。该步报错发生在engine = builder.build_cuda_engine(network)
- 可能原因：tensorRT只支持float32, float, int32, 但不支持int64所以如果模型采用了int64则需要替代该数据类型.
  参考解答：https://devtalk.nvidia.com/default/topic/1045064/tensorrt/onnx-and-tensorrt-error-network-must-have-at-least-one-output/
  在tensorRT中，针对onnx模型中的数值，以及shape输出，都默认会基于pytorch变成long()，但long()并不被tensorRT支持，所以导致报错。需要用int(x.size(0))把结果从long转化成int32  
  解决办法：把cls_head, bbox_head的view()中的常量都用int()进行强制指定，即out = out.view(int(out.size(0)), int(-1), int(self.num_classes))
  

### 关于硬件jetson nano

1. 硬件安装和软件烧录：
    - 核心板部分基本都集成了，就是在安装无线wifi卡的时候，需要把核心板从母版上拆下来(参考https://cloud.tencent.com/developer/article/1421906)
      同是安装wifi卡前，需要把橙色保护膜剪开个缺口把螺丝固定处漏出来。(更简单的方式是采用那种360随身usb wifi即可)
    - 系统安装很简单，只需要把nvidia的镜像文件烧录到一个新的microSD卡里边即可，烧录工具采用开源的Etcher。
    - 然后通电，就能点亮屏幕了。如果使用的是小型7寸屏，需要从母板usb额外提供一根电源线给显示屏，同时母板上J48需要增加短路帽。


2. 启动摄像头：jetson nano支持的摄像头有2种，一种usb类型，另一种是树莓派CSI排线接口的摄像头(但必须是IMX219 sensor，比如Raspberry Pi Camera Module v2)
参考：https://mp.weixin.qq.com/s?__biz=MjM5NTE3Nzk4MQ==&mid=2651234579&idx=1&sn=7f10f030e9c60b15c6805fa1ea495347&chksm=bd0e75818a79fc977693c16d7eb4dd87709eab82542687208d5ba34cfcc877a2da68ab6a106b&scene=21#wechat_redirect
摄像头排线安装好以后，检查linux系统是否识别出来的方法是终端运行：ls /dev/vid*, 如果显示了/dev/video0就说明已经识别出来了。
此时就可以通过opencv等各种方式启动摄像头了。

2. 终端启动CSI摄像头： 参考https://www.yahboom.com/build.html?id=2504&cid=301
    - nvgstcapture-1.0 启动
    - nvgstcapture-1.0 --prev-res=3 表示预览视频的分辨率、高度、宽度，其中的数值是从2-12，每个数值代表一个宽高分辨率
    - nvgstcapture-1.0 --cus-prev-res=1920x1080 表示自定义预览视频分辨率
    - q + 回车 表示关闭摄像头
    - j + 回车 表示捕获图片

3. 代码启动摄像头：