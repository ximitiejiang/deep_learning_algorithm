# Simple_ssd_pytorch

Originally i want to implement a simplifier SSD detector in this repo, but now it has already integrated some other one-stage detector algorithms, including SSD, M2det, RetinaNet.
<div align=center><img src="https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/test14_result.jpeg"/></div>
<div align=center><img src="https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/test11_result.jpg"/></div>
<div align=center><img src="https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/video_result.jpg"/></div>

the ssd detector implementation is simplified from [mmdetection](https://github.com/open-mmlab/mmdetection)

### Features
curently it support VGG16 backbone, the pretrained weights is from caffe on [here](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth).
besides this, other features include:
+ support multi-GPU training on parallel mode
+ support cpu/gpu nms
+ support training on coco/voc daaset
+ mean precision data to be update soon

![model structure](https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/ssd_structure.jpg)

some details for the model:
+ input img size: (300,300) or (500,500)
+ vgg16 output with extra layers: (vgg/L22: (512,38,38)), (vgg/L34: (1024,19,19)),
(extra/1: (512,10,10)), (extra/3: (256,5,5)), (extra/5: (256,3,3)), (extra/1: (256,1,1))
+ base anchors: base size is get from ratios of anchor with img, this ratios is a prior knowledge in article.
and 2 types of scales(1, sqrt(max_size/min_size)), and 3 or 5 types of ratios(1,2,1/2) or(1,2,1/2,3,1/3), 
total base anchors for each featmaps are (4,6,6,6,4,4), a trick here is ratio_major, means sort ratio first.
if 4 base anchors means `(2,1)*(1,3)[:4]`, and if 6 base anchors means `(2,1)*(1,5)[:6]`
+ total anchors: `38*38*4 + 19*19*6 + 10*10*6 + 5*5*6 + 3*3*4 + 1*1*4 = 8732`
+ grid anchors: 
+ anchor targets: assign anchors with ious(-1 means no_use, 0 means bg, 1~n means fg), but no anchor sampling applied.
so in order to avoid inbalance of fg and bg, hard negtive mining method is applied in loss calculate stage. 
+ bbox classify: using weighted smooth l1 loss func
+ bbox coordinate regression: using weighted cross_entropy loss func
+ as hard negtive mining was used on training,so no nms using on training, 
but using nms on testing imgs in ssd head module, nms or soft_nms can be chosen.

### Installation(test enviroments)
+ pytorch 0.4.1, cudn9.0, cython, mmcv
+ git clone this repo
+ create data folder in repo root and create symlink to your own dataset source documents.
+ run the compile.sh for recompiling nms.(if not success you can manually delete exist cpp/so files in nms document)
+ add repo root to sys path (or run requriements.py to temporarily add repo root to sys path).
+ the necessary document structure is as below:

```
simple_ssd_pytorch
├── config
├── data
├── dataset
├── model
├── utils
├── weights
├── README.md
├── requirements.py
├── compile.sh
├── TEST_ssd_coco.py
├── TEST_ssd_img.py
├── TRAIN_ssd_coco.py
├── TRAIN_ssd_voc.py
```

### train & test
+ run train_xxx.py for training on specific dataset, currently support voc07/12 and coco2017
+ run test_xxx_img.py for testing one pic based on trained model.
+ run test_xxx_xxx.py for evaluating the model mAP on specific dataset.

### mAP results

Note: using repeatdataset to load dataset 10 times for each epoch, which means below epoch num should be 10 times than conventional epoch num. For example, below epoch 6 means No.60 epoch for a dataset.

+ benchmarking

    [from here](https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md)    
    + mAP = 0.778 (mmdet SSD300-vgg)
    + mAP = 0.804 (mmdet SSD512-vgg)
    
    [from here](https://github.com/qijiezhao/pytorch-ssd)
    + mAP = 0.805 (m2det300-vgg) 
    + mAP = 0.821 (m2det512-vgg)
    + mAP = 0.827 (m2det512-resnet101)
    
+ train setting 1(SSD300): 2 imgs per GPU, 2 workers per GPU, SGD lr = 2e-4, momentum=0.9, weight_decay=5e-4
    + mAP = 0.348 (epoch 3)
    + mAP = 0.507 (epoch 6)
    + mAP = 0.649 (epoch 12)
    + mAP = 0.721 (epoch 24)
```
+-------------+------+-------+--------+-----------+-------+
| class       | gts  | dets  | recall | precision | ap    |
+-------------+------+-------+--------+-----------+-------+
| aeroplane   | 285  | 3471  | 0.919  | 0.076     | 0.792 |
| bicycle     | 337  | 2553  | 0.938  | 0.125     | 0.823 |
| bird        | 459  | 16264 | 0.869  | 0.025     | 0.641 |
| boat        | 263  | 14427 | 0.935  | 0.017     | 0.665 |
| bottle      | 469  | 25656 | 0.793  | 0.015     | 0.421 |
| bus         | 213  | 2600  | 0.958  | 0.079     | 0.811 |
| car         | 1201 | 17527 | 0.962  | 0.067     | 0.851 |
| cat         | 358  | 2695  | 0.941  | 0.125     | 0.819 |
| chair       | 756  | 46359 | 0.923  | 0.015     | 0.525 |
| cow         | 244  | 1969  | 0.959  | 0.122     | 0.760 |
| diningtable | 206  | 5201  | 0.932  | 0.037     | 0.697 |
| dog         | 489  | 3588  | 0.930  | 0.127     | 0.768 |
| horse       | 348  | 1916  | 0.937  | 0.172     | 0.825 |
| motorbike   | 325  | 1979  | 0.942  | 0.156     | 0.815 |
| person      | 4528 | 90185 | 0.939  | 0.047     | 0.762 |
| pottedplant | 480  | 34010 | 0.835  | 0.012     | 0.424 |
| sheep       | 242  | 2770  | 0.905  | 0.080     | 0.696 |
| sofa        | 239  | 2249  | 0.954  | 0.107     | 0.770 |
| train       | 282  | 3323  | 0.975  | 0.083     | 0.847 |
| tvmonitor   | 308  | 8813  | 0.909  | 0.032     | 0.699 |
+-------------+------+-------+--------+-----------+-------+
| mAP         |      |       |        |           | 0.721 |
+-------------+------+-------+--------+-----------+-------+
```
+ training setting 2(SSD300): 4 imgs per GPU, 2 workers per GPU, SGD lr = 2e-4, momentum=0.9, weight_decay=5e-4  (add batch size)
    + mAP = 0.695 (epoch 3)
    + mAP = 0.721 (epoch 6)
    + mAP = 0.750 (epoch 12)
    + mAP = 0.774 (epoch 24)
    + insight: bigger batch size can improve the mAP
```
+-------------+------+-------+--------+-----------+-------+
| class       | gts  | dets  | recall | precision | ap    |
+-------------+------+-------+--------+-----------+-------+
| aeroplane   | 285  | 1420  | 0.909  | 0.184     | 0.819 |
| bicycle     | 337  | 1359  | 0.911  | 0.229     | 0.845 |
| bird        | 459  | 3459  | 0.893  | 0.120     | 0.742 |
| boat        | 263  | 3520  | 0.901  | 0.069     | 0.737 |
| bottle      | 469  | 6781  | 0.783  | 0.055     | 0.506 |
| bus         | 213  | 1068  | 0.958  | 0.194     | 0.848 |
| car         | 1201 | 6527  | 0.952  | 0.180     | 0.862 |
| cat         | 358  | 1264  | 0.936  | 0.267     | 0.873 |
| chair       | 756  | 12013 | 0.862  | 0.056     | 0.605 |
| cow         | 244  | 1073  | 0.951  | 0.227     | 0.831 |
| diningtable | 206  | 1557  | 0.932  | 0.129     | 0.734 |
| dog         | 489  | 1903  | 0.939  | 0.244     | 0.847 |
| horse       | 348  | 1111  | 0.943  | 0.301     | 0.864 |
| motorbike   | 325  | 1152  | 0.935  | 0.269     | 0.839 |
| person      | 4528 | 29187 | 0.924  | 0.145     | 0.792 |
| pottedplant | 480  | 7516  | 0.788  | 0.051     | 0.518 |
| sheep       | 242  | 1424  | 0.913  | 0.157     | 0.789 |
| sofa        | 239  | 1151  | 0.941  | 0.216     | 0.786 |
| train       | 282  | 1077  | 0.940  | 0.249     | 0.863 |
| tvmonitor   | 308  | 2842  | 0.929  | 0.102     | 0.770 |
+-------------+------+-------+--------+-----------+-------+
| mAP         |      |       |        |           | 0.774 |
+-------------+------+-------+--------+-----------+-------+
```

+ training setting 3(SSD512): 4 imgs per GPU, 2 workers per GPU, SGD lr = 1e-3, momentum=0.9, weight_decay=5e-4, adding MLFPN neck and with bigger lr
    + batch size analysis: trainset=5011(voc07)+11540(voc12), totally 16551(07+12), so after 10 times repeatdataset, the total length of dataset is 165510
    and if 2GPUs and 4 pics per GPU, batch size should be 8, so the iter_num = 165510/8 = 20689(20688.75) 
    + bigger lr issue(lr from 2e-4 to 1e-3): lead to fast decrease of the loss, but also lead to stable status after 3 epoches. 
    + mAP = 0.385 (epoch 4, with warmup lr and base lr= 0.001), the lr is too big, so decrease to 0.0002
    + mAP = 0.580 (epoch 8, with step lr = 0.0002), not too much improvement for the loss when decrease the lr
    + mAP = 0.570 (epoch 9, with step lr = 0.001), no improvement or even worse loss by adding num_img_per_gpu from 4 to 8, and lr increase to 0.001 which is same with original paper(0.002 for 4 GPU, here use 0.001 for 2 GPU) 
    + mAP = 0.664 (epoch 12, with step lr = 0.0002), loss decrease again from 3.7 to 3.2 by decrease lr back to 0.0002 with 8 imgs per GPU, comparing with epoch 8, batch img increase helps(4 to 8), lr increase not helps.
    + mAP = 0.680 (epoch 16, with step lr = 0.0002), loss decrease from 3.2 to 2.9 by using lr=0.0002 util epoch 20, which is same as original paper(4gpu/lr0.0004 util epoch20)
    + mAP = 0.706 (epoch 20, with step lr = 0.0002), by using lr=0.0002
    + mAP = 0.xxx (epoch 24, with step lr = 0.00002), by using lr=0.00002
    + mAP =  (epoch 28, with step lr = 0.000002), by using lr=0.000002
      
### Todo
+ [x] support eval on coco dataset
+ [x] support eval on voc
+ [x] add MLFPN in the module lib
+ [ ] add RetinaNet detector
+ [ ] support distributed training