from utils.version import get_versions
import numpy as np

def nms_numpy(boxes):
    """
    args:
        boxes (array (n, 5))
    return:
        keep (array, (n,))
    """
    box = boxes[:, :4]
    scores = boxes[:,4]
    inds = boxes.argsort()[::-1]
    while inds.size() > 1:
        pass


def check_list():
    
    l1 = []
    l1.append(1)
    l1.extend([2,3])
    l1.insert(-1, 200)
    
def check_random():
    a = np.random.randn(2,4)
    print(a)

def check_dict():
    d = dict(a= 1, b =2)
    print(d)
    
    d1 = dict(a=2, b=3)
    d2 = dict(c=4,d=5)
    d = dict(**d1, **d2)
    print(d)

def check_set():
    a = {1,3,4,3,1}
    b = {3,1,2,4,2}
    print(a)
    c = a | b
    print(c)
    
def sort():
    np.random.seed(12)
    a = np.random.randn(2,4)
    print(a)
    c = [i * 3 for i in range(3)]
    print(c)
    
    b = np.array([4,7,3,6,4,9,1])
    b = sorted(b)
    d = argsort(b)
    print(b)
    print(d)

if __name__ == "__main__":
    check_list()
    check_random()
    check_dict()
    check_set()
    sort()