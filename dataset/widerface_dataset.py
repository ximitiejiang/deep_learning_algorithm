#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:44:16 2019

@author: ubuntu
"""
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
    def __init__(self,                  
                 root_path=None,
                 ann_file=None,
                 subset_path=None,
                 img_transform=None,
                 label_transform=None,
                 bbox_transform=None,
                 aug_transform=None,
                 data_type=None):
        super().__init__(
                root_path, 
                ann_file, 
                subset_path,
                img_transform, 
                label_transform, 
                bbox_transform,
                aug_transform,
                data_type)
    
    def load_annotation_inds(self, ann_file):
        """从多个标注文件读取标注列表
        """
        img_anns = []
        for i, af in enumerate(ann_file): 
            with open(af) as f:
                img_ids = f.readlines()
            for j in range(len(img_ids)):
                img_ids[j] = img_ids[j][:-1]  # 去除最后的\n字符
            # 基于图片id打开annotation文件，获取img/xml文件名
            for img_id in img_ids:
                idx = int(img_id.split('_')[0])
                folder = self.folders[idx] if idx != 61 else self.folders[idx - 1]
#                tmp = img_id.split('_')[:-2]
#                names = []
#                for t in tmp:
#                    if not t in names and not t[0].islower():  # 如果字符没有重复出现，没有小写字母开头,则存入，否则直接退出for循环
#                        names.append(t)
#                    else:
#                        break                tmp = img_id.split('_')[:-2]
#                names = []
#                for t in tmp:
#                    if not t in names and not t[0].islower():  # 如果字符没有重复出现，没有小写字母开头,则存入，否则直接退出for循环
#                        names.append(t)
#                    else:
#                        break
#                folder = names[0] + '--' + names[1]   # 需要额外生成文件夹名称
#                if len(names) > 2:
#                    for k in range(2, len(names)):    # 获取部分特殊文件名中的folder
#                        folder += '_' + names[k]
#                folder = names[0] + '--' + names[1]   # 需要额外生成文件夹名称
#                if len(names) > 2:
#                    for k in range(2, len(names)):    # 获取部分特殊文件名中的folder
#                        folder += '_' + names[k]
                
                img_file = self.subset_path[i] + 'images/{}/{}.jpg'.format(folder, img_id)
                xml_file = self.subset_path[i] + 'Annotations/{}.xml'.format(img_id)
                
                img_anns.append(dict(img_id=img_id, img_file=img_file, xml_file=xml_file))
        return img_anns
    
    
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