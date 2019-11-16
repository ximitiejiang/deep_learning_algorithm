#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:17:33 2019

@author: ubuntu
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from model.bbox_regression_lib import delta2bbox
from model.nms_lib import nms_wrapper
from model.bbox_head.ssd_head import get_base_anchor_params
from model.anchor_generator_lib import AnchorGenerator

# SSD后处理器
class PostprocessorSSD():
    def __init__(self, cfg):
        # 生成base_anchor所需参数
        n_featmap = len(cfg.head.in_channels)
        base_sizes, scales, ratios, centers = \
            get_base_anchor_params(cfg.head.input_size, cfg.head.anchor_size_ratio_range, 
                                   n_featmap, cfg.head.anchor_strides, cfg.head.anchor_ratios)
        # 创建anchor生成器
        self.anchor_generators = []
        for i in range(n_featmap):
            anchor_generator = AnchorGenerator(base_sizes[i], scales[i], 
                                               ratios[i], ctr=centers[i],    
                                               scale_major=False) 
            # 保留的anchor: 2*3的前4个(0-3), 2*5的前6个(0-5)
            keep_anchor_indices = range(0, len(ratios[i])+1)
            anchor_generator.base_anchors = anchor_generator.base_anchors[keep_anchor_indices]
            self.anchor_generators.append(anchor_generator)
        self.anchors = []
        for i in range(len(self.featmap_sizes)):
            self.anchors.append(self.anchor_generators[i].grid_anchors(
                    self.featmap_sizes[i], self.anchor_strides[i]))   
        self.anchors = torch.cat(self.anchors, dim=0)
            
    
    def process(self, outputs, img_metas):
        """outputs为模型输出: (1,8732, 21), (1, 8732, 4)
        后处理需要注意的问题：
        1. 需要获得featmap_size来生成anchors，但featmap_size对于部分可变图片的来说是动态在forward时计算得到的，这里需要根据下采样比例手动计算
        2. TODO: 后处理是否要完全脱离torch和tensor，而完全在numpy的条件下进行?
        """
        cls_scores = outputs[0][0]
        bbox_preds = outputs[1][0]
        img_metas = img_metas[0]
        device = cls_scores.device
        
        img_size = img_metas['pad_shape']

        # 计算每张图的bbox预测
        scale_factor = img_metas['scale_factor']        
        cls_scores = F.softmax(cls_scores, dim=1) # 概率化
        bbox_preds = delta2bbox(self.anchors, bbox_preds, self.target_means, self.target_stds, img_size) # 坐标化     
        bboxes_preds = bbox_preds / bbox_preds.new_tensor(scale_factor[:4])  # 相对原图的尺寸
        # nms
        bboxes, labels, _ = nms_wrapper(bboxes_preds, cls_scores, **self.cfg.nms) # (n_cls,)(m,5),  (n_cls,)(m,),  (n_cls,)(m,5,2) 
        
        # 把按类别的数据合并(这里不需要按类别，只有在evaluation才需要)      
        labels = np.concatenate(labels, axis=0) - 1  # (m, ) 恢复到0为起点
        bboxes = np.concatenate(bboxes, axis=0)  # (m,5)
        scores = bboxes[:, -1]
        
        return bboxes, labels, scores
    
# YOLO后处理器
class PostprocessorYOLO(object):
    """Class for post-processing the three outputs tensors from YOLOv3-608."""

    def __init__(self, cfg):
#                 yolo_masks,
#                 yolo_anchors,
#                 obj_threshold,
#                 nms_threshold,
#                 yolo_input_resolution,
#                 num_categories):
        """Initialize with all values that will be kept when processing several frames.
        Assuming 3 outputs of the network in the case of (large) YOLOv3.

        Keyword arguments:
        yolo_masks -- a list of 3 three-dimensional tuples for the YOLO masks
        yolo_anchors -- a list of 9 two-dimensional tuples for the YOLO anchors
        object_threshold -- threshold for object coverage, float value between 0 and 1
        nms_threshold -- threshold for non-max suppression algorithm,
        float value between 0 and 1
        input_resolution_yolo -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        params = cfg.postprocessor.params
        self.masks = params.yolo_masks
        self.anchors = params.yolo_anchors
        self.object_threshold = params.obj_threshold
        self.nms_threshold = params.nms_threshold
        self.input_resolution_yolo = params.yolo_input_resolution
        self.num_categories = params.num_categories

    def process(self, outputs, resolution_raw):
        """Take the YOLOv3 outputs generated from a TensorRT forward pass, post-process them
        and return a list of bounding boxes for detected object together with their category
        and their confidences in separate lists.

        Keyword arguments:
        outputs -- outputs from a TensorRT engine in NCHW format
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        """
        outputs_reshaped = list()
        for output in outputs:
            outputs_reshaped.append(self._reshape_output(output))

        boxes, categories, confidences = self._process_yolo_output(
            outputs_reshaped, resolution_raw)
        # 调整bbox的形式从(x,y,w,h)到(xmin,ymin,xmax,ymax)
        boxes[:,2:] = boxes[:, 2:] + boxes[:, :2]
        boxes = boxes.astype(np.int32)
        return boxes, categories, confidences

    def _reshape_output(self, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (height,width,3,85) dimensionality after further reshaping.

        Keyword argument:
        output -- an output from a TensorRT engine after inference
        """
        output = np.transpose(output, [0, 2, 3, 1])
        _, height, width, _ = output.shape
        dim1, dim2 = height, width
        dim3 = 3
        # There are CATEGORY_NUM=80 object categories:
        dim4 = (4 + 1 + self.num_categories)
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def _process_yolo_output(self, outputs_reshaped, resolution_raw):
        """Take in a list of three reshaped YOLO outputs in (height,width,3,85) shape and return
        return a list of bounding boxes for detected object together with their category and their
        confidences in separate lists.

        Keyword arguments:
        outputs_reshaped -- list of three reshaped YOLO outputs as NumPy arrays
        with shape (height,width,3,85)
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        """

        # E.g. in YOLOv3-608, there are three output tensors, which we associate with their
        # respective masks. Then we iterate through all output-mask pairs and generate candidates
        # for bounding boxes, their corresponding category predictions and their confidences:
        boxes, categories, confidences = list(), list(), list()
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence = self._process_feats(output, mask)
            box, category, confidence = self._filter_boxes(box, category, confidence)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)

        boxes = np.concatenate(boxes)
        categories = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # Scale boxes back to original image shape:
        width, height = resolution_raw
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        # Using the candidates from the previous (loop) step, we apply the non-max suppression
        # algorithm that clusters adjacent bounding boxes to a single bounding box:
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self._nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return None, None, None

        boxes = np.concatenate(nms_boxes)
        categories = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)

        return boxes, categories, confidences

    def _process_feats(self, output_reshaped, mask):
        """Take in a reshaped YOLO output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.

        Keyword arguments:
        output_reshaped -- reshaped YOLO output as NumPy arrays with shape (height,width,3,85)
        mask -- 2-dimensional tuple with mask specification for this output
        """

        # Two in-line functions required for calculating the bounding box
        # descriptors:
        def sigmoid(value):
            """Return the sigmoid of the input."""
            return 1.0 / (1.0 + math.exp(-value))

        def exponential(value):
            """Return the exponential of the input."""
            return math.exp(value)

        # Vectorized calculation of above two functions:
        sigmoid_v = np.vectorize(sigmoid)
        exponential_v = np.vectorize(exponential)

        grid_h, grid_w, _, _ = output_reshaped.shape

        anchors = [self.anchors[i] for i in mask]

        # Reshape to N, height, width, num_anchors, box_params:
        anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])
        box_xy = sigmoid_v(output_reshaped[..., :2])
        box_wh = exponential_v(output_reshaped[..., 2:4]) * anchors_tensor
        box_confidence = sigmoid_v(output_reshaped[..., 4])

        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = sigmoid_v(output_reshaped[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.input_resolution_yolo
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        # boxes: centroids, box_confidence: confidence level, box_class_probs:
        # class confidence
        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Take in the unfiltered bounding box descriptors and discard each cell
        whose score is lower than the object threshold set during class initialization.

        Keyword arguments:
        boxes -- bounding box coordinates with shape (height,width,3,4); 4 for
        x,y,height,width coordinates of the boxes
        box_confidences -- bounding box confidences with shape (height,width,3,1); 1 for as
        confidence scalar per element
        box_class_probs -- class probabilities with shape (height,width,3,CATEGORY_NUM)

        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.object_threshold)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).

        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep

# 
postprocessor_dict={
        'ssd': PostprocessorSSD,
        'yolov3':PostprocessorYOLO}


def get_postprocessor(cfg):
    cls_name = cfg.postprocessor.type
    cls_type = postprocessor_dict[cls_name]
    return cls_type(cfg)
