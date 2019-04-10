"""采用mlfpn neck配合m2det head实现m2det算法
"""
# model settings
input_size = 512   # 图片尺寸加大
model = dict(
    type='SingleStageDetector',
    pretrained='open-mmlab://vgg16_caffe',
    backbone=dict(
        type='M2detVGG',
        input_size=input_size,
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        l2_norm_scale=20),
    neck=dict(
        type='MLFPN',
        backbone_type='M2detVGG',
        input_size=input_size,
        planes=256,
        smooth=True,
        num_levels=8,
        num_scales=6,
        side_channel=512,
        sfam=False,
        compress_ratio=16),
    bbox_head=dict(
        type='M2detHead',
        input_size=input_size,
        planes = 256,
        num_levels = 8,
        num_classes = 21,  # 21 for VOC, 81 for COCO
        anchor_strides=(8, 16, 32, 64, 128, 256),      # 6 layers的anchor的base size定义
        size_pattern = [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        size_featmaps = [(64,64), (32,32), (16,16), (8,8), (4,4), (2,2)],
        anchor_ratio_range = ([2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2)))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# model training and testing settings
# dataset settings
dataset_type = 'VOCDataset'
data_root = './data/VOCdevkit/'    # 位置调整
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
data = dict(
    imgs_per_gpu=8,    # 从4改为8看看效果（4是mmdetection的一般设置，但m2det上来就是16/32之类的大batch size）
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            img_scale=(512, 512),      # 增大图片尺寸
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 4)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
            resize_keep_ratio=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(512, 512),     # 增大图片尺寸
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(512, 512),   # 增大图片尺寸
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='SGD', lr=2e-4, momentum=0.9, weight_decay=5e-4)   # 从8块GPU的1e-3变为2e-4 (前4个epoch忘了改，loss降很快且到第4个epoch就平稳了)
optimizer_config = dict()
# learning policy
#lr_config = dict(
#    policy='step',
#    warmup='linear',
#    warmup_iters=500,
#    warmup_ratio=1.0 / 3,
#    step=[16, 20])
lr_config = dict(
    policy='list',    # 修改学习率更新hook，自定义更新lr
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    epoch_list=[4, 8, 16, 22],  # 更新epoch
    lr_list=[0.0002, 0.0002, 0.0002, 0.0004]) # 更新lr
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
#        dict(type='TensorboardLoggerHook')   # 没什么必要，直接用text绘图即可
    ])
# yapf:enable
# runtime settings
gpus=2
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/m2det512_voc'
load_from = None
#resume_from = None
resume_from = './work_dirs/m2det512_voc/latest.pth'
workflow = [('train', 1)]
