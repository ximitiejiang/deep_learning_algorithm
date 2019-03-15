#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo 'cleaning the documents before compiling...'
cd mmdet/ops/roi_align
if [ -d "build" ]; then
    rm -r build
fi

cd ../nms
make clean
make PYTHON=${PYTHON}
