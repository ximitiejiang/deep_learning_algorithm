#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:09:00 2019

@author: ubuntu

多线程下载一个文件，参考http://blog.topspeedsnail.com/archives/8462
"""

import sys
import requests
import threading
import datetime
 
# 传入的命令行参数，要下载文件的url
#url = sys.argv[1]
 
 
def Handler(start, end, url, filename):
    
    headers = {'Range': 'bytes=%d-%d' % (start, end)}
    r = requests.get(url, headers=headers, stream=True)
    
    # 写入文件对应位置
    with open(filename, "r+b") as fp:
        fp.seek(start)
        var = fp.tell()
        fp.write(r.content)
 
 
def download_file(url, num_thread = 5):
    start = datetime.datetime.now().replace(microsecond=0) 
    r = requests.head(url)
    try:
        file_name = url.split('/')[-1]
        file_size = int(r.headers['content-length'])   # Content-Length获得文件主体的大小，当http服务器使用Connection:keep-alive时，不支持Content-Length
    except:
        print("检查URL，或不支持对线程下载")
        return
 
    #  创建一个和要下载文件一样大小的文件
    fp = open(file_name, "wb")
    fp.truncate(file_size)
    fp.close()
 
    # 启动多线程写文件
    part = file_size // num_thread  # 如果不能整除，最后一块应该多几个字节
    for i in range(num_thread):
        start = part * i
        if i == num_thread - 1:   # 最后一块
            end = file_size
        else:
            end = start + part
 
        t = threading.Thread(target=Handler, kwargs={'start': start, 'end': end, 'url': url, 'filename': file_name})
        t.setDaemon(True)
        t.start()
 
    # 等待所有线程下载完成
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
        t.join()
    end = datetime.datetime.now().replace(microsecond=0)
    print('%s finished download with %f seconds' % (file_name, end-start))
 
if __name__ == '__main__':
    url = "https://anaconda.org/conda-forge/opencv/4.1.0/download/linux-64/opencv-4.1.0-py37h79d2e43_0.tar.bz2"
    download_file(url, num_thread=5)
