#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:39:29 2019

@author: ubuntu
"""
import numpy as np

class RandomFSCropTransform(object):
    """Random fixed size crop随机固定尺寸切割: 同时影响img/bbox
    专门用于traffic sign recognize contest的图像变换类"""
    def __init__(self, req_sizes):
        self.req_sizes = req_sizes   # (h, w) 
        self.req_w = self.req_sizes[0]
        self.req_h = self.req_sizes[1]
    
    def __call__(self, img, bbox, label):
        """
        Args:
            img(array): (h,w,c)
            bbox(list): (1,)(4,) coordinate of (xmin,ymin,xmax,ymax)
        """
#        img_new = np.zeros((self.req_h, self.req_w, 3))   # (h,w,c) = (800,1333,3)
#        h, w, c = img.shape
#        xmin = int(np.random.choice(np.arange(0,bbox[0][0])))  
#        ymin = int(np.random.choice(np.arange(0,bbox[0][1])))
#        xmax = xmin + self.req_w
#        ymax = ymin + self.req_h
#        if xmax >= w :
#            xmax = w - 1
#        if ymax >= h:
#            ymax = h - 1
        xmin,ymin,xmax,ymax = self.get_crop_coord(img, bbox)

        img_new = img.transpose(2,0,1)[:, ymin:ymax, xmin:xmax].transpose(1,2,0)  # (h,w,c) -> (c,h,w) ->crop -> (h,w,c)
        
        bbox_xmin = bbox[0][0] - xmin
        bbox_ymin = bbox[0][1] - ymin
        bbox_xmax = bbox[0][2] - xmin
        bbox_ymax = bbox[0][3] - ymin
        
        bbox_new = [[bbox_xmin, bbox_ymin,
                    bbox_xmax, bbox_ymax]]
        bbox_new = np.array(bbox_new)
        label_new = label
        
        return img_new, bbox_new, label_new
            
        
    def get_crop_coord(self, img, bbox):
        h, w, _ = img.shape
        xmin = bbox[0][0]
        ymin = bbox[0][1]
        xmax = bbox[0][2]
        ymax = bbox[0][3]
        while True:
            if xmin*ymin < (w - xmax)*(h - ymax):
                crop_xmin = np.random.choice(np.arange(0,xmin))
                crop_ymin = np.random.choice(np.arange(0,ymin))
                crop_xmax = crop_xmin + self.req_w
                crop_ymax = crop_ymin + self.req_h
                if crop_xmax >= w or crop_ymax >= h or crop_xmax < xmax or crop_ymax < ymax:
                    continue
                else:
                    return (int(crop_xmin), int(crop_ymin), int(crop_xmax), int(crop_ymax))
            else:
                crop_xmax = np.random.choice(np.arange(xmax,w))
                crop_ymax = np.random.choice(np.arange(ymax,h))
                crop_xmin = crop_xmax - self.req_w
                crop_ymin = crop_ymax - self.req_h
                if crop_xmin < 0 or crop_ymin < 0 or crop_xmin > xmin or crop_ymin > ymin:
                    continue
                else:
                    return (int(crop_xmin), int(crop_ymin), int(crop_xmax), int(crop_ymax))
            
            