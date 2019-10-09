# Deep Learning Algorithms
<div align=center><img src="https://github.com/ximitiejiang/deep_learning_algorithm/blob/master/test/nba_epoch9.png"/></div>

### 主要内容

本仓库主要用于深度学习算法和训练框架的重实现，主要参考：mmdetection, awesome-semantic-segmentation-pytorch

1. 物体检测相关
    - SSD
    - RetinaNet
    - FCOS
    - FasterRcnn
    - MaskRcnn
    
2. 语义分割相关
    - FCN
    - DeeplabV3
    - PSPNet
    
3. 物体跟踪相关
    - tbf

4. 训练框架
    - Runner
    
5. 支持的数据集
    - VOC07+12
    - COCO2017
    - WIDERFACE
    - CITYSCAPE
    
### 安装

```
cd model/nms
python setup.py build_ext --inplace

cd ..
cd ..
python setup.py
```

### 注意事项

1. 关于添加到系统文件夹
<br>每次重启电脑或重启终端，都需要运行python setup.py文件把根目录加到系统文件夹中才能进行相关训练或预测。
也可以永久添加根目录，参考setup.py注释。

2. 关于归一化
<br>如果是使用pytorch的预训练模型，那么对应的mean, std都是<1的数据，也就需要先normalize(value/255，设置norm=True)，然后再进行标准化到标准正态分布。
如果是使用caffe的预训练模型，那么对应std=1

3. 关于文件夹位置
<br>需要预先设置工作文件夹work_dir等文件夹，并且确保文件夹存在，程序有文件夹检查功能，以防训练好以后模型无法保存。

4. 关于config中的task
<br>需要设置成classifier, segmentator, detector之一，用于区分是哪种任务。

5. 关于dataloader中的collate_fn
<br>其中的dict_collate可以支持对img/seg数据的堆叠，也支持输入图片尺寸不同的情况，会自动进行padding统一到一个batch的最大图片尺寸。

6. 关于anchor的尺寸
<br>可以沿用某些算法中anchor的设置比例，也可以通过聚类(参考bbox_kmean)来计算针对某一数据集的anchor个数和大小，确保在anchor个数越少越好的前提下获得足够大的平均iou。

7. 关于新增模型
<br>如果自定义新的模型，需要把新增模型在prepare_training的model zoo区域导入并添加进对应的model dict中用于调用。

8. 关于数据集的设置
<br>WIDERFACE数据集的annotation文件采用变换过的voc格式，参考https://github.com/sovrasov/wider-face-pascal-voc-annotations.git

9. 关于预训练模型
<br>检测模型的预训练backbone权重来自mmdetection(包含一部分pytorch的模型和一部分caffe的模型)
<br>分割模型的预训练backbone权重来自gluon
