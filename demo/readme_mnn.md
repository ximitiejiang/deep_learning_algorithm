# MNN

### MNN安装
1. 安装非常简单，一步安装graphviz，一步安装mnn
参考：https://www.yuque.com/mnn/en/usage_in_python
```
apt-get install graphviz
pip install -U MNN
```

### MNN基本命令行命令
1. mnn          # 列出mnn的所有命令行指令(就这几条)

2. mnnops       # 所有mnn算子

3. mnnconvert   # mnn用来转换其他模型到mnn模型
- 这个转换器使用：
```
# ONNX to MNN
./MNNConvert -f ONNX --modelFile xxx.onnx --MNNModel xxx.mnn --bizCode biz  # 这是官方写法直接报错。要改成如下
mnnconvert -f ONNX --modelFile xxx.onnx --MNNModel xxx.mnn --bizCode biz  # 依然报错：可能无法兼容onnx的版本(我的版本是ir version 6)

#Pytorch to MNN: 
#先把pytorch模型转换成mnn模型，然后
./MNNConvert -f ONNX --modelFile xxx.onnx --MNNModel xxx.mnn --bizCode biz

```

4. mnnquant     # 量化mnn模型

5. mnnvisual    # 模型的逻辑保存成image(可视化)

