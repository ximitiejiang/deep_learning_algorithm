# Simple_ssd_pytorch

a simplifier ssd detector implement in pytorch
![test_img](https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/test14_result.jpeg)
![test_img](https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/test13_result.png)
this ssd implementation is simplified from [mmdetection](https://github.com/open-mmlab/mmdetection)

### features
curently it support VGG16 backbone, the pretrained weights is from caffe on [here]('https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth').
besides this, other features include:
+ support multi-GPU training on parallel mode
+ support cpu/gpu nms
+ support training on coco/voc daaset.

### installation(test enviroments)
+ pytorch 0.4.1, cudn9.0, cython, mmcv
+ git clone this repo
+ create data folder in repo root and create symlink to your own dataset source documents.
+ run the compile.sh for recompiling nms.(if not success you can manually delete exist cpp/so files in nms document)
+ add repo root to sys path (or run requriements.py to temporarily add repo root to sys path).
+ the necessary document structure is as below:
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


### train & test
+ run train_xxx.py for training on specific dataset, currently support voc07/12 and coco2017
+ run test_xxx_img.py for testing one pic based on trained model.
+ run test_xxx_xxx.py for evaluating the model mAP on specific dataset.

### TODO
+ [x] support eval on coco dataset
+ [ ] support eval on voc
+ [ ] support distributed training
