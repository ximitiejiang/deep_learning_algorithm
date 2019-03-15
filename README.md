# Simple_ssd_pytorch

a simplifier ssd detector implement in pytorch
![test_img](https://github.com/ximitiejiang/simple_ssd_pytorch/blob/master/data/test14_result.jpeg)
this ssd implement is simplified from [mmdetection](https://github.com/open-mmlab/mmdetection)

### features
curently it support multi-GPU training on parallel mode, support cpu/gpu nms, and support training on coco
voc daaset.

### installation(test enviroments)
+ pytorch 0.4.1, cudn9.0, cython, mmcv
+ git clone this repo
+ create data folder in repo root and create symlink to your own dataset source documents.
+ recompile the build.py for nms.(before compile you may need to delete exist cpp/so files)
+ add repo root to sys path (or run requriements.py to temporarily add repo root to sys path).

### train & test
+ run train_xxx.py for training on specific dataset, currently support voc07/12 and coco2017
+ run test_xxx_img.py for testing one pic based on trained model.
+ run test_xxx_xxx.py for evaluating the model mAP on specific dataset.

### TODO
+ [x] support eval on coco dataset
+ [ ] support eval on voc
+ [ ] support distributed training
