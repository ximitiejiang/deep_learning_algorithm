#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo 'cleaning the documents before compiling...'
cd utils/nms
if [ -d "build" ]; then
    rm -r build
fi

make clean
make PYTHON=${PYTHON}

echo 'successfully compile!'
