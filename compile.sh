#!/usr/bin/env bash

# 用这个shell脚本文件可以直接编译nms模块文件

PYTHON=${PYTHON:-"python"}

echo 'cleaning the documents before compiling...'
cd utils/nms                      # 进入nms文件目录
if [ -d "build" ]; then           # 如果存在build文件夹，则删除该文件夹
    rm -r build
fi

make clean                        # 运行makefile里边的clean分支：删除.so/.cpp文件 
make PYTHON=${PYTHON}             # 运行makefile里边的all分支：调用setup.py文件，进行编译

echo 'successfully compile!'
