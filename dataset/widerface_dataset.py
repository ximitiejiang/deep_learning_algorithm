#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:44:16 2019

@author: ubuntu
"""
import numpy as np
from dataset.voc_dataset import VOCDataset

class WIDERFaceDataset(VOCDataset):
    """wider face数据集，来自港中文大学
    训练集图片总数：12880，跟voc07+12的数据量差不多
    原始下载数据集不能直接使用，需要采用转换成voc格式的xml标注文件，参考：
    https://github.com/open-mmlab/mmdetection/tree/master/configs/wider_face
    https://github.com/sovrasov/wider-face-pascal-voc-annotations
    """
    CLASSES = ('face', )
    
    folders = ['0--Parade', '1--Handshaking', '2--Demonstration','3--Riot', '4--Dancing',
               '5--Car_Accident', '6--Funeral','7--Cheering', '8--Election_Campain', '9--Press_Conference',
               '10--People_Marching', '11--Meeting', '12--Group', '13--Interview', '14--Traffic', '15--Stock_Market', 
               '16--Award_Ceremony', '17--Ceremony', '18--Concerts', '19--Couple',  
               '20--Family_Group', '21--Festival', '22--Picnic', '23--Shoppers', '24--Soldier_Firing','25--Soldier_Patrol', 
               '26--Soldier_Drilling', '27--Spa', '28--Sports_Fan', '29--Students_Schoolkids', 
               '30--Surgeons', '31--Waiter_Waitress', '32--Worker_Laborer', '33--Running', '34--Baseball', 
               '35--Basketball', '36--Football', '37--Soccer', '38--Tennis', '39--Ice_Skating', '40--Gymnastics',
               '41--Swimming', '42--Car_Racing', '43--Row_Boat', '44--Aerobics', '45--Balloonist', '46--Jockey', 
               '47--Matador_Bullfighter', '48--Parachutist_Paratrooper', '49--Greeting', 
               '50--Celebration_Or_Party', '51--Dresses', '52--Photographers', '53--Raid', '54--Rescue', 
               '55--Sports_Coach_Trainer', '56--Voter', '57--Angler', '58--Hockey', '59--people--driving--car', 
               '61--Street_Battle', ]
    def __init__(self, *args, **kwargs):
        # 增加landmark_file作为ann_file的替代
        super().__init__( *args, **kwargs)

        
    def load_annotations(self, ann_file):
        """从多个标注文件读取标注列表
        """
        print('Loading widerface dataset...')
        img_anns = []
        # 如果有landmark_file, 则基于landmark_file生成img_anns
        if self.landmark_file is not None:
            for i, af in enumerate(self.landmark_file):
                with open(af) as f:
                    img_infos = f.readlines()
                    for j in range(len(img_infos)):
                        img_infos[j] = img_infos[j][:-1]  # 去除最后的\n字符
                    
                    for img_info in img_infos:
                        if img_info.startswith('#'):  # 只要检测到#就是一个新的img
                            path = img_info[2:]
                            img_file = self.img_prefix[i] + 'images/' +path
                            img_anns.append(dict(img_file=img_file, labels=[]))
                        else:
                            label = img_info.split(' ')
                            label = [float(lab) for lab in label]
                            img_anns[-1]['labels'].append(label)
        # 如果没有landmark_file, 则基于原有的ann_file加载img_anns            
        else:
            for i, af in enumerate(ann_file): 
                with open(af) as f:
                    img_ids = f.readlines()
                    for j in range(len(img_ids)):
                        img_ids[j] = img_ids[j][:-1]  # 去除最后的\n字符
                    # 基于图片id打开annotation文件，获取img/xml文件名
                    for img_id in img_ids:
                        idx = int(img_id.split('_')[0])
                        folder = self.folders[idx] if idx != 61 else self.folders[idx - 1]
        
                        
                        img_file = self.img_prefix[i] + 'images/{}/{}.jpg'.format(folder, img_id)
                        xml_file = self.img_prefix[i] + 'Annotations/{}.xml'.format(img_id)
                        
                        img_anns.append(dict(img_id=img_id, img_file=img_file, xml_file=xml_file))
        return img_anns
    
    def parse_ann_info(self, idx):
        """如果有landmark_file，则采用该parse_ann_info(), 否则沿用父类VOC的parse_ann_info()
        label_infos的格式说明：每组label info为20个数值，前4个值为x0,y0,w,h, 后15个值为以0间隔的5个点坐标x,y, 最后一个值???
        如果该bbox没有landmark, 则所有landmark点为-1，否则间隔点为1或者0
        """
        label_infos = self.img_anns[idx]['labels']
        bboxes = []
        labels = []
        landmarks = []
        for idx, info in enumerate(label_infos):
            # 提取bbox
            xmin = info[0]
            ymin = info[1]
            xmax = info[0] + info[2]
            ymax = info[1] + info[3]
            bboxes.append([xmin, ymin, xmax, ymax])
            # 提取landmark
            point0 = [info[4], info[5]]
            point1 = [info[7], info[8]]
            point2 = [info[10], info[11]]
            point3 = [info[13], info[14]]
            point4 = [info[16], info[17]]
            landmarks.append(np.array([point0, point1, point2, point3, point4]))
            # 提取labels
            if point0[0] < 0:     # 如果没有landmark, 则标签定为-1
                labels.append(-1)
            else:
                labels.append(1)# 如果没有landmark, 则标签定为1
        
        return dict(bboxes=np.array(bboxes).astype(np.float32), 
                    labels=np.array(labels).astype(np.int64), 
                    landmarks=np.array(landmarks).astype(np.float32))
        
        
if __name__ == '__main__':
    data_root_path = '/home/ubuntu/MyDatasets/WIDERFace/'
    params = dict(
                    root_path = data_root_path,
                    ann_file = [data_root_path + 'train.txt'],
                    subset_path = [data_root_path + 'WIDER_train/'],
                    data_type='train')
    
    dset = WIDERFaceDataset(**params)
    data = dset[0]
    print(len(dset))