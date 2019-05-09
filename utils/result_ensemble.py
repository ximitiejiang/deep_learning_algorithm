#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:43:39 2019

@author: suliang

本部分用于对多模型进行集成，增强结果的精度：采用vote的方式进行结果集成
参考：https://github.com/MLWave/Kaggle-Ensemble-Guide/blob/master/src/kaggle_vote.py

基础知识
1. sys.argv 表示从命令行获得形参输入，类似于argparse模块
   可以理解为sys.argv代表了外部输入的一个参数list，用list切片方式获得参数，list的每个元素都可以是一个子list
    + sys.argv[0]获得的是程序本身
    + sys.argv[1]获得的是第1个外部参数
    + sys.argv[1:] 获得的是外部参数的列表list(也是从第1个外部参数开始)

2. re是正则表达式模块
    + pat = re.compile(pattern)用来生成一个正则表达式对象，
    + data = pat.match(some_text) 用来
    + data = pat.search(some_text) 用来
    + data = pat.split(some_text) 用来
    + data.group(2)表示？？
    + 也可以用通用的一种更直观的方式写成： re.split('正则表达式', some_text)，从而省略compile()函数并且2步并一步
    + 实例: some_text = 'a,b,,,,,,c d'
            re.split('[,]+', some_text)

3. glob.glob/glob.iglob模块，用于文件搜索，支持通配符
    + 常用通配符: 'dir/*'表示下层同级所有文件和子文件夹，'dir/*/*'表示下面2层所有子文件和子文件夹，
                  'dir/submi?.txt'表示用?代表单个字符通配符
                  'dir/*[0-9].*' 表示用[]代表单个字符的范围
    + file_list = glob.glob(r'../*.py') 获得相对父路径下所有py文件
    + file_list = glob.glob(r'/home/ubuntu/suliang_git/simple_ssd_pytorch/*.py') 获得绝对路径下所有.py文件
    + file_gen = glob.iglob(r'../*.py') 生成迭代器对象
    
4. 要区别开正则表达式和通配符
    + 正则表达式通常用在文本过滤
    + 通配符通常用在文件名搜索
"""

from collections import defaultdict, Counter
from glob import glob
import sys
import re

def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):
    pattern = re.compile(r"(.)*_[w|W](\d*)_[.]*")        # 创建正则表达式对象
    if method == "average":            
        scores = defaultdict(list)
        
    with open(loc_outfile,"w") as outfile:
    #weight_list may be usefull using a different method
        weight_list = [1]*len(glob(glob_files))              # 假定输入是3个文件，则这里weight_list=[1,1,1]
        for i, glob_file in enumerate( glob(glob_files) ):   # 循环调用没一个输入文件
            print("parsing: {}".format(glob_file))
            if weights == "weighted":
                weight = pattern.match(glob_file)     # 在输入文件中搜索
                if weight and weight.group(2):
                    print("Using weight: {}".format(weight.group(2)))
                    weight_list[i] = weight_list[i]*int(weight.group(2))
                else:
                    print("Using weight: 1")
              # sort glob_file by first column, ignoring the first line
            lines = open(glob_file).readlines()
            lines = [lines[0]] + sorted(lines[1:])
            for e, line in enumerate( lines ):
                if i == 0 and e == 0:
                    outfile.write(line)
                if e > 0:
                    row = line.strip().split(",")
                    for l in range(1,weight_list[i]+1):
                        scores[(e,row[0])].append(row[1])
        for j,k in sorted(scores):
            outfile.write("%s,%s\n"%(k,Counter(scores[(j,k)]).most_common(1)[0][0]))
        print("wrote to {}".format(loc_outfile))


if __name__ == '__main__':
#    glob_files = sys.argv[1]        # 这里glob_files可以是多个结果文件名比如空格间隔输入"a.csv" "b.csv"，或通配符输入"submit*.csv"
#    loc_outfile = sys.argv[2]       # 输出文件
    
    glob_files = []
    loc_outfile = []
    weights_strategy = "uniform"    # 默认权重策略是uniform，还可以是average
    if len(sys.argv) == 4:
        weights_strategy = sys.argv[3]  # 如果还有第4个参数，则是权重策略
        
    kaggle_bag(glob_files, loc_outfile, weights=weights_strategy)
