#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:47:36 2019

@author: ubuntu
"""
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2
from utils.colors import COLORS

# %%
def vis_loss_acc(buffer_dict, title='result: '):
    """可视化结果: 至少包含acc(比如验证)
    输入: dict[key, value_list]
            loss(list): [loss1, loss2,..] or [[iter1, loss1], [iter2, loss2], ...]
            acc(list): [acc1, acc2,..] or [[iter1, acc1], [iter2, acc2], ...]
    """
    accs = None
    losses = None
    lrs = None
    if buffer_dict.get('acc', None) is not None:
        accs = buffer_dict['acc']
    if buffer_dict.get('loss', None) is not None:
        losses = buffer_dict['loss']
    if buffer_dict.get('lr', None) is not None:
        lrs = buffer_dict['lr']
        
    if title is None:
        prefix = ""
    else:
        prefix = title
    
    
    if isinstance(accs[0], list) or isinstance(accs[0], tuple):  # 如果losses列表里边包含idx
        x = np.array(accs)[:,0]
        y_acc = np.array(accs)[:,1]
    else:  # 如果losses列表里边不包含idx只是单纯loss数值
        x = np.arange(len(accs))
        y_acc = np.array(accs)
    # 绘制acc
    prefix += ' accs'
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title(prefix)
    ax1.set_ylabel('acc')
    lines = ax1.plot(x,y_acc, 'r', label='acc')
    # 绘制loss
    if losses is not None and len(losses) > 0:
        if isinstance(losses[0], list) or isinstance(losses[0], tuple):
            x = np.array(losses)[:,0]
            y_loss = np.array(losses)[:,1]
        else:
            x = np.arange(len(losses))
            y_loss = np.array(losses)
        prefix += ' losses'
        ax1.set_title(prefix)
        ax2 = ax1.twinx()
        ax2.set_ylabel('loss')
        l2 = ax2.plot(x, y_loss, 'g', label='loss')
        lines += l2
        
    # 提取合并的legend
    legs = [l.get_label() for l in lines]     
    # 显示合并的legend
    ax1.legend(lines, legs, loc=0)
    plt.grid()
    plt.show()
    
    # 由于量纲问题，lr单独绘制
    if lrs is not None and len(lrs) > 0:
        if isinstance(lrs[0], list) or isinstance(lrs[0], tuple):
            x = np.array(lrs)[:,0]
            y_lr = np.array(lrs)[:,1]
        else:
            x = np.arange(len(lrs))
            y_lr = np.array(lrs)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_title(title + ' lr')
        ax1.set_ylabel('lr')
        lines = ax1.plot(x,y_lr, 'r', label='lr')
        legs = [l.get_label() for l in lines]   
        ax1.legend(lines, legs, loc=0)
        plt.grid()
        plt.show()


# %%
def vis_img_bbox(img, bboxes, labels, class_names=None,
        thickness=1, font_scale=0.5):
    """简化版显示img,bboxes,labels(无法筛选score置信度)
    img: (h,w,c)
    bboxes: (m, 4)
    labels: (m, )
    """
    from utils.colors import COLORS
    # 准备颜色
    color_list = []
    for color in COLORS.values():
        color_list.append(color)
    color_list.pop(-1) # the last one is white, reserve for text only, not for bboxes
    color_list = color_list * 12  # 循环加长到84，可以显示80类的coco
    random_colors = np.stack(color_list, axis=0)  # (7,3)
    # 开始绘制
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
            cv2.FONT_HERSHEY_DUPLEX, font_scale, [255,255,255])
    cv2.imshow('result', img)
    return img


def vis_bbox(bboxes, img=None):
    """绘制一组bboxes, (n,4) or (n,2): 
    如果是2-points模式，输入bboxes为(xmin,ymin,xmax,ymax)
    如果是wh模式，输入bboxes为(w, h)
    """
    if img is None:
        img = 255 * np.zeros((500, 500, 3)).astype(np.uint8)  # 必须设置成3通道，否则无法显示颜色的线条；必须设置成uint8否则无法被opencv显示。
    x_ctr = img.shape[1] / 2
    y_ctr = img.shape[0] / 2
    
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.numpy()
    bboxes = bboxes.astype(np.int32)
    # 如果是4点模式，则绘制bbox在真实位置
    if bboxes.shape[1] == 4:
        for bbox in bboxes:
            left_top = (bbox[0], bbox[1])
            right_bottom = (bbox[2], bbox[3])
            cv2.rectangle(img, left_top, right_bottom, (0,255,0), thickness=1)
    # 如果是宽高模式，则绘制bbox在图片中心
    elif bboxes.shape[1] == 2:  
        for bbox in bboxes:
            left_top = (int(x_ctr - bbox[0] / 2), int(y_ctr - bbox[1] / 2))
            right_bottom = (int(x_ctr + bbox[0] / 2), int(y_ctr + bbox[1] / 2))
            cv2.rectangle(img, left_top, right_bottom, [0, 255, 0], thickness=1)
    cv2.imshow('bboxes', img)
    return img


def vis_all_opencv(img, bboxes, scores, labels, class_names=None, score_thr=0, 
                    instance_colors=None, thickness=1, font_scale=0.6,
                    show=True, win_name='cam', wait_time=0, saveto=None): # 如果输出到文件中则指定路径
    """采用opencv作为底层显示img/bbox/labels
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
    return img


def vis_all_pyplot(img, bboxes, scores=None, labels=None, class_names=None, score_thr=0, 
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



def vis_dataset_one_class(dataset, class_name, saveto=None, show=None):
    """显示一个数据集中某一个类的所有图片：先生成带bbox的图片，然后拼接图片成一张
    可用来检查某一类图片的总体特征，比如小物体类型的比例。
    """
    pass
  
    
# %%

def vis_cam(src, predictor, class_names=None, score_thr=None):
    """用于对摄像头数据进行检测
    args:
        src: int(表示cam_id) or str(表示video文件路径)
        predictor: 表示预测计算器，用来创建模型，计算显示需要的输出(img, bboxes, scores, labels)
    """
    # 如果是int则为cam_id, 如果是str则为video path
    if isinstance(src, (int, str)):
        if isinstance(src, int):
            cam_id = src
            capture = cv2.VideoCapture(cam_id)
        elif isinstance(src, str):
            capture = cv2.VideoCapture(src)
        assert capture.isOpened(), 'Cannot capture source'
    # 循环预测
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret, img = capture.read()
        # 检查是否正确返回
        if not ret:  # failure to read or run to end frame
            cv2.destroyAllWindows()
            capture.release()
            break
        # 检查是否有按键中断
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
        # 返回迭代器
        for results in predictor(img):
            vis_all_opencv(*results, class_names, score_thr)
                


# %%
def vis_activation_hist(source):
    """用于查看激活层输出值的分布：
    参考：deep learning from scratch， p178
    激活层的输出一般称之为激活值，代表了特征在前向计算过程中是否正常，
    激活值如果集中在左右两侧，则说明有经过激活函数后取值会越来越大，可能产生梯度爆炸或者梯度消失。
    激活值如果集中在中间，则说明激活分布有偏向，在输出表现力上受限，模型学习能力就不够。
    所以激活值应该在+-1之前区域较广泛分布，才是比较合理。
    
    使用方法：datalist.append(x)
    
    Args:
        data_list(list): 表示激活函数输出的每一层的值，[d1, d2,..]每个元素为(b,c,h,w)
    """
    from utils.tools import loadvar
    if isinstance(source, str) and os.path.isfile(source):  # 如果传入一个文件路径，则打开
        # TODO
        data = loadvar()
    elif isinstance(source, list):
        data = source
    else:
        raise ValueError('source should be a path or a list.')
    # 开始绘图
    plt.figure()
    for i, li in enumerate(data):  # 提取每层
        plt.subplot(2, len(data)/2+1, i+1)  # 2行
        plt.title(str(i+1)+"-layer")  
        plt.hist(li.flatten(), 30, range=(-3,3))  # 展平成(b*c*h*w,), 然后取30个区间, 由于有bn，所以只统计取值在中间的数。
    plt.show()
    
    
# %%
from tqdm import tqdm
def vis_dataset_bbox_area(cfg_path):
    """用于统计一个数据集的所有bbox的面积值，以及对bbox的w,h进行：
    """    
    from utils.prepare_training import get_config, get_dataset
    cfg = get_config(cfg_path)
    cfg.trainset.params.ann_file = [cfg.trainset.params.ann_file[0]]   # 先只用voc07
    
    trainset = get_dataset(cfg.trainset, cfg.transform)
    class_names = trainset.CLASSES
    
    ws  = []
    hs = []
    areas = []
    labels = []
    for data in tqdm(trainset):
        img_meta = data['img_meta']
        gt_labels = data['gt_labels']
        gt_bboxes = data['gt_bboxes']
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        area = w * h
        
        ws.extend(w)
        hs.extend(h)
        areas.extend(area)
        labels.extend(gt_labels)    
        
    ws = np.array([w.item() for w in ws])   # (k,)
    hs = np.array([h.item() for h in hs])   # (k,)
    areas = np.array([area.item() for area in areas])   # (k,)
    labels = np.array([label.item() for label in labels]) # (k,)
    
    # 先绘制总的分布图
    plt.figure()
    plt.title('all')
    plt.hist(areas, 30, range=(0,90000))
    plt.show()
    
    # 再分别绘制每个类的hist
    plt.figure()
    for class_id in range(1, 21):  # 假定20类
        inds = labels == class_id
        class_areas = areas[inds]
        plt.subplot(4, 5, class_id)
        plt.title(class_names[class_id - 1])
        plt.hist(class_areas, 30, range=(0, 90000))
    plt.show()
    
    # 然后计算size = sqrt(area), 绘制size的scatter
    plt.figure()
    plt.title('w and h scatter')
    plt.scatter(ws, hs)
    
    # 然后对横坐标w,纵坐标h的size尺寸做聚类
    data = np.concatenate([ws[:, None], hs[:, None]], axis=1)
    centers = kmean(data, k=5)
    plt.scatter(centers[:, 0], centers[:, 1], s=50, c='r')
    
    plt.show()
    
    

def kmean(data, k):
    """进行二维数据的聚类分析
    args:
        data: (m,2)分别是w,h，所以是二维坐标上的聚类
        k: 聚类个数
    return
        centers: (k,2) k个聚类中心坐标
    """
    from sklearn.cluster import KMeans
    _kmean = KMeans(n_clusters = k)
    _kmean.fit(data)
    centers = _kmean.cluster_centers_
    return centers
    

if __name__ == '__main__':
    bboxes = np.array([[20,30], [50,70], [100,200],[200,300]])
    
    vis_bbox(bboxes)    
    