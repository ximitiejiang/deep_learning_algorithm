# 用这个makefile调用setup没有成功
# 改成直接build setup成功了： python3 setup.py build_ext --inplace

PYTHON=${PYTHON:-python}

all:
	echo "Compiling nms kernels..."
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -f *.so
