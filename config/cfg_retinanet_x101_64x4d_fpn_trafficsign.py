# model settings
model = dict(
    type='RetinaNet',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,      # resnext101
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=22,   # traffic sign = 22(总计21类0-20，加背景就是22类),  voc using 21, coco using 81
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    smoothl1_beta=0.11,
    gamma=2.0,
    alpha=0.25,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'TrafficSign'    # 改成trafficsign
data_root = './data/traffic_sign/'  # 改成trafficsign
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # 采用的是pytorch的模型，但mean/std还是用的caffe的？？？
data = dict(           # repeatdataset不加了，在coco训练12epoch，voc上调成24
    imgs_per_gpu=2,    # retinanet的图片尺寸更大，所以每个gpu只放2张图片, 如果4张就CUDA out of memory了
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_label_fix.csv',
        img_prefix=data_root + 'Train_fix/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        with_label=True,
        extra_aug=None),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train_label_fix.csv',
        img_prefix=data_root + 'Train_fix/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        with_label=True,
        extra_aug=None),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'train_label_fix.csv',
        img_prefix=data_root + 'Test_fix/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        with_label=False,
        extra_aug=None,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)  # 学习率调小到原来8块GPU的1/4(0.01 to 0.002)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))          # 增加了一个梯度截断？
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[10, 16, 20])    # [8,12,20] with 2 imgs per gpu, change to [10,16,20] with 4 imgs per gpu
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
gpus = 2
total_epochs = 20   # coco不加repeatdataset训练了12轮，对应voc从12加倍到24轮
device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/retinanet_trafficsign'
load_from = None
resume_from = './work_dirs/retinanet_trafficsign/latest.pth'
#resume_from = None
workflow = [('train', 1)]
