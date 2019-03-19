# Simple_ssd_pytorch

This is a simplifier ssd detector implement in pytorch, base document on [here](https://arxiv.org/pdf/1512.02325.pdf)
![test_img](https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/test14_result.jpeg)
![test_img](https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/test11_result.jpg)
![video_img](https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/video_result.jpg)

this ssd implementation is simplified from [mmdetection](https://github.com/open-mmlab/mmdetection)

### features
curently it support VGG16 backbone, the pretrained weights is from caffe on [here](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth).
besides this, other features include:
+ support multi-GPU training on parallel mode
+ support cpu/gpu nms
+ support training on coco/voc daaset
+ mean precision data to be update soon
![model structure](https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/ssd.jpg)

some details for the model:
+ input img size: (300,300) or (500,500)
+ vgg16 output with extra layers: (vgg/L22: (512,38,38)), (vgg/L34: (1024,19,19)),
(extra/1: (512,10,10)), (extra/3: (256,5,5)), (extra/5: (256,3,3)), (extra/1: (256,1,1))
+ base anchors: base size is get from ratios of anchor with img, this ratios is a prior knowledge in article.
and 2 types of scales(1, sqrt(max_size/min_size)), and 3 or 5 types of ratios(1,2,1/2) or(1,2,1/2,3,1/3), 
total base anchors for each featmaps are (4,6,6,6,4,4), a trick here is ratio_major, means sort ratio first.
if 4 base anchors means (2,1)*(1,3)[:4], and if 6 base anchors means (2,1)*(1,5)[:6]
+ total anchors: `38*38*4 + 19*19*6 + 10*10*6 + 5*5*6 + 3*3*4 + 1*1*4 = 8732`
+ grid anchors: 
+ anchor targets: assign anchors with ious(-1 means no_use, 0 means bg, 1~n means fg), but no anchor sampling applied.
so in order to avoid inbalance of fg and bg, hard negtive mining method is applied in loss calculate stage. 
+ bbox classify: using weighted smooth l1 loss func
+ bbox coordinate regression: using weighted cross_entropy loss func
+ as hard negtive mining was used on training,so no nms using on training, 
but using nms on testing imgs in ssd head module, nms or soft_nms can be chosen.

### installation(test enviroments)
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

### TODO
+ [x] support eval on coco dataset
+ [ ] support eval on voc
+ [ ] support distributed training
