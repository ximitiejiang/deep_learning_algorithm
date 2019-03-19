#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:01:48 2019

@author: ubuntu
"""
import cv2
import os

def frames2video(frame_dir,
                 video_file,
                 fps=30,
                 fourcc='XVID',
                 filename_tmpl='{:06d}.jpg',
                 start=0,
                 end=0,
                 show_progress=True):
    """Read the frame images from a directory and join them as a video

    Args:
        frame_dir (str): The directory containing video frames.
        video_file (str): Output filename.
        fps (int): FPS of the output video.
        fourcc (str): Fourcc of the output video, this should be compatible
            with the output file type.
        filename_tmpl (str): Filename template with the index as the variable.
        start (int): Starting frame index.
        end (int): Ending frame index.
        show_progress (bool): Whether to show a progress bar.
    """
    if end == 0:
        ext = filename_tmpl.split('.')[-1]
        end = len([name for name in scandir(frame_dir, ext)])
    first_file = osp.join(frame_dir, filename_tmpl.format(start))
#    check_file_exist(first_file, 'The start frame not found: ' + first_file)
    img = cv2.imread(first_file)
    height, width = img.shape[:2]
    resolution = (width, height)
    vwriter = cv2.VideoWriter(video_file, VideoWriter_fourcc(*fourcc), fps,
                              resolution)

    def write_frame(file_idx):
        filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
        img = cv2.imread(filename)
        vwriter.write(img)

#    if show_progress:
#        track_progress(write_frame, range(start, end))
    for i in range(start, end):
        filename = os.path.join(frame_dir, filename_tmpl.format(i))
        img = cv2.imread(filename)
        vwriter.write(img)
    vwriter.release()


def video_record():
    """cv2.VideoWriter_fourcc(**fourcc), 
    fourcc='I420'表示为压缩的YUV编码，4:2:0色度子采样
    fourcc='PIM1'
    fourcc='XVID'表示为MPEG-4编码，文件扩展名.avi
    fourcc='THEO'表示为Ogg Vorbis, 文件扩展名.ogv
    fourcc='FLV1'表示为Flash视频，文件扩展名为.flv
    """
    capture = cv2.VideoCapture(0) 
    fps = capture.get(cv2.CAP_PROP_FPS) 
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) # (w,h)
    videoWriter = cv2.VideoWriter('./record.avi', 
                                  cv2.VideoWriter_fourcc('I', '4', '2', '0'), 
                                  fps, 
                                  size) 
    success, frame = capture.read() 
    numFrame = 5 * fps -1 
    while success and numFrame > 0: 
        videoWriter.write(frame) 
        success, frame = capture.read() 
        numFrame -= 1 
        
    capture.release()
    
    
    
    
    
    
    