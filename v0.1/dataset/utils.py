import copy
import numpy as np
import cv2
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from matplotlib import pyplot as plt
from dataset.color_transforms import color2value, COLORS

def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """把tensor数据逆变换成可以显示img数据：逆rgb化，逆chw化，逆归一化，逆tensor化
    """
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)  # to cpu, to numpy, chw to hwc
#        img = mmcv.imdenormalize(
#            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        # TODO: 逆归一化后是否缺了整数化和clip到0-255
        img = (img * std) + mean                               # denorm
        if to_rgb:
            img = img[...,[2,1,0]]                             # rgb2bgr
        imgs.append(np.ascontiguousarray(img))
    return imgs

    
def opencv_vis_bbox(img, bboxes, labels, scores, score_thr=0, class_names=None, 
                    instance_colors=None, thickness=1, font_scale=0.6,
                    show=True, win_name='cam', wait_time=0, saveto=None): # 如果输出到文件中则指定路径
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
#    img = imread(img)

    if score_thr > 0:
#        assert bboxes.shape[1] == 5
#        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    
    color_list = []
    for color in COLORS.values():
        color_list.append(color)
    color_list.pop(-1) # the last one is white, reserve for text only, not for bboxes
    random_colors = np.stack(color_list, axis=0)  # (7,3)
    random_colors = np.tile(random_colors, (12,1))[:len(class_names),:]  # (84,3) -> (20,3)or(80,3)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(                  # 画方框
            img, left_top, right_bottom, random_colors[label].tolist(), 
            thickness=thickness)
        label_text = class_names[       # 准备文字
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += ': {:.02f}'.format(bbox[-1])
            
        txt_w, txt_h = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness = 1)[0]
        cv2.rectangle(                  # 画文字底色方框
            img, (bbox_int[0], bbox_int[1]), 
            (bbox_int[0] + txt_w, bbox_int[1] - txt_h - 4), 
            random_colors[label].tolist(), -1)  # -1为填充，正整数为边框thickness
        cv2.putText(
            img, label_text, (bbox_int[0], bbox_int[1] - 2),     # 字体选择cv2.FONT_HERSHEY_DUPLEX, 比cv2.FONT_HERSHEY_COMPLEX好一点
            cv2.FONT_HERSHEY_DUPLEX, font_scale, [255,255,255])  # 字体白色
        
    if saveto is not None:
        cv2.imwrite(saveto, img)
    if show:
        cv2.imshow(win_name, img)


def vis_bbox(img, bboxes, labels=None, scores=None, score_thr=0, class_names=None,
             instance_colors=None, alpha=1., linewidth=1.5, ax=None, saveto=None):
    """另外一个图片+bbox显示的代码
    注意，该img输入为hwc/bgr(因为在test环节用这种格式较多)，如果在train等环节使用，
    就需要把img先从chw/rgb转成hwc/bgr
    Args:
        img (ndarray): (h,w,c), BGR and the range of its value is
            :math:`[0, 255]`. If this is :obj:`None`, no image is displayed.
        bbox (ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(x_{min}, y_{min}, x_{max}, y_{max})` in the second axis.
        label (ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`class_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        score_thr(float): A float in (0, 1), bboxes scores with lower than
            score_thr will be skipped. if 0 means all bboxes will be shown.
        class_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        instance_colors (iterable of tuples): List of colors.
            Each color is RGB format and the range of its values is
            :math:`[0, 255]`. The :obj:`i`-th element is the color used
            to visualize the :obj:`i`-th instance.
            If :obj:`instance_colors` is :obj:`None`, the red is used for
            all boxes.
        alpha (float): The value which determines transparency of the
            bounding boxes. The range of this value is :math:`[0, 1]`.
        linewidth (float): The thickness of the edges of the bounding boxes.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of label must be same as that of bbox')
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of score must be same as that of bbox')
    
    if score_thr > 0:                      # 只显示置信度大于阀值的bbox
        score_id = scores > score_thr
        # 获得scores, bboxes
        scores = scores[score_id]
        bboxes = bboxes[score_id]
        labels = labels[score_id]
        
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if img is not None:
        img = img[...,[2,1,0]]         # hwc/bgr to rgb
        ax.imshow(img.astype(np.uint8))
    # If there is no bounding box to display, visualize the image and exit.
    if len(bboxes) == 0:
        return ax
    
    # instance_colors可以等于list: [255,0,0]
    # 也可等于None
    # 否则等于随机7种颜色之一
    color_list = []
    for color in COLORS.values():
        color_list.append(color)
    color_list.pop(-1) # 去除白色，用于文字颜色
    random_colors = np.stack(color_list, axis=0)  # (7,3)
    if class_names is None:
        color_len =1
    else:
        color_len = len(class_names)
    random_colors = np.tile(random_colors, (12,1))[:color_len,:]  # (84,3) -> (20,3)or(80,3)
    
    if instance_colors is None:
        instance_colors = random_colors
#        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
#        instance_colors[:, 0] = 255
    else:
        assert len(instance_colors) == 3, 'instance_colors should be a list [n1,n2,n3].'
        instance_colors = np.tile(instance_colors, (color_len, 1))
#    instance_colors = np.array(instance_colors)

    for i, bb in enumerate(bboxes):        # xyxy to xywh
        xy = (bb[0], bb[1])
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
        # 先定义默认颜色        
        color = instance_colors[0] / 255  # 默认用第一个颜色红色[255,0,255]作为常规显示图片和bbox, 但要归一到（0-1）来表示颜色
        
        caption = []
        if labels is not None and class_names is not None:
            lb = labels[i]
            caption.append(class_names[lb])
            color = instance_colors[lb] /255     # 如果有标签，则按标签类别定义颜色
        if scores is not None:
            sc = scores[i]
            caption.append('{:.2f}'.format(sc))
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha))

        if len(caption) > 0:
            ax.text(bb[0], bb[1]-2,     # 改到左下角点(xmin,ymin,xmax,ymax) ->(xmin,ymax)
                    ': '.join(caption),
                    style='italic',
                    color = 'white',  # 默认是白色字体
                    bbox={'facecolor': color, 'alpha': 0.5, 'pad': 0}) 
                    #文字底色跟边框颜色一样，透明度=1表示不透明，边空1
    if saveto is not None:
        plt.savefig(saveto)
    return ax


class RepeatDataset(object):

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len


class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)


def get_dataset(data_cfg, dataset_class):
    """"获得数据集
    Args:
        data_cfg(dict): 存放所有数据集初始化参数的字典cfg.data.train
        dataset_class(class): 数据集的类
    Return:
        dset(obj): 生成的数据集
    """
    if data_cfg['type'] == 'RepeatDataset':
        return RepeatDataset(
            get_dataset(data_cfg['dataset'], dataset_class), data_cfg['times'])

    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset
    assert len(proposal_files) == num_dset

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)   # 需要深拷贝，是因为后边会pop()操作避免修改了原始cfg
        data_info['ann_file'] = ann_files[i]
        data_info['proposal_file'] = proposal_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        
        data_info.pop('type')
        dset = dataset_class(**data_info)  # 弹出type字段，剩下就是数据集的参数
        dsets.append(dset)
    if len(dsets) > 1:
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset

""" 如下是为了在test dataset进程中创建dataloder的子程序, 待review
"""
from functools import partial
#from mmcv.runner import get_dist_info
#from mmcv.parallel import collate
#from .sampler import GroupSampler, DistributedGroupSampler
from model.parallel.collate import collate
from torch.utils.data import DataLoader
from dataset.sampler import GroupSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=False,
                     **kwargs):
#    if dist:
#        rank, world_size = get_dist_info()
#        sampler = DistributedGroupSampler(dataset, imgs_per_gpu, world_size,
#                                          rank)
#        batch_size = imgs_per_gpu
#        num_workers = workers_per_gpu
#    else:
    if not dist:
        if not kwargs.get('shuffle', True):
            sampler = None
        else:
            sampler = GroupSampler(dataset, imgs_per_gpu)
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
                             pin_memory=False,
                             **kwargs)

    return data_loader



if __name__ == '__main__':
    
    source = 'coco'
    
    if source == 'voc':   
        from voc_dataset import VOCDataset
        data_root = '../data/VOCdevkit/'  # 指代ssd目录下的data目录
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
        cfg_train=dict(
            type='RepeatDataset',
            times=10,
            dataset=dict(
                type='VOCDataset',
                ann_file=[
                    data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                    data_root + 'VOC2012/ImageSets/Main/trainval.txt'
                ],
                img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
                img_scale=(300, 300),
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
                resize_keep_ratio=False))
        
        trainset = get_dataset(cfg_train, VOCDataset)
        classes = trainset.CLASSES
        data = trainset[1120]  # dict('img', 'img_meta', )
        """已做的数据处理：rgb化，chw化，归一化，tensor化"""
        bbox = data['gt_bboxes'].data.numpy()
        label = data['gt_labels'].data.numpy()
        img = data['img'].data.numpy()     # 逆tensor
        img1 = img.transpose(1,2,0)   # 逆chw
        img2 = np.clip((img1 * img_norm_cfg['std'] + img_norm_cfg['mean']).astype(np.int32), 0, 255)  # 逆归一
        vis_bbox(img2[...,[2,0,1]], bbox, label-1, class_names=classes)  # vis_bbox内部会bgr转rgb，所以这里要用bgr输入
    
    if source == 'coco':
        from dataset.coco_dataset import CocoDataset
        data_root = '../data/coco/'
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
        cfg_train=dict(
            type='RepeatDataset',
            times=5,
            dataset=dict(
                type='CocoDataset',
                ann_file=data_root + 'annotations/instances_train2017.json',
                img_prefix=data_root + 'train2017/',
                img_scale=(300, 300),
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
                resize_keep_ratio=False))
    
        trainset = get_dataset(cfg_train, CocoDataset)
        classes = trainset.CLASSES
        data = trainset[1000]
        """已做的数据处理：rgb化，chw化，归一化，tensor化"""
        bbox = data['gt_bboxes'].data.numpy()
        label = data['gt_labels'].data.numpy()
        img = data['img'].data.numpy()     # 逆tensor
        img1 = img.transpose(1,2,0)   # 逆chw
        img2 = np.clip((img1 * img_norm_cfg['std'] + img_norm_cfg['mean']).astype(np.int32), 0, 255)  # 逆归一
        vis_bbox(img2[...,[2,0,1]], bbox, label-1, class_names=classes)  # vis_bbox内部会bgr转rgb，所以这里要用bgr输入
        
        
