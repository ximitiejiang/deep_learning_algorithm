# Deep Learning Algorithms

这个代码仓库基本算是重新写了一遍mmdetection和mmcv，为什么要重新造轮子？主要有2个原因，第一个是为了自己更好的
理解算法的细节，另一方面也是为了后续修改了优化的时候更方便操作。

mmdetection本身真的是一个写的非常好，并且内容非常丰富的物体检测框架，那这个代码仓库的侧重点主要是在：
1. 尽可能简化整个架构，让主线更清晰更容易理解。
2. 减少了一些封装，让一些细节暴露出来，便于理解和调试。
3. 采用一些更简单的实现替代当前比较复杂的实现。
4. 增加详细中文注释

有得就有失，简化实现以后，理解起来是简单点，但通用性下降了一些，也可能有一些还没发现的bug。
但build from scratch所获得的对细节的理解是宝贵的。


### 算法总成测试
1. 所有算法测试代码和配置文件都在demo文件夹中


### 子程序测试
1. 一些重要子程序测试文件都在test文件夹中


@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}