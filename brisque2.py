import cv2
# from libsvm.python.svmutil import *
# import libsvm.python.svmutil as svmutil 
from brisque import BRISQUE
from brisq.Python.libsvm.python import brisquequality as b
import time
import imquality.brisque as brisque
import PIL.Image
from multiprocessing import Pool

def multipleBrisque(imagesArray):
    brisq = BRISQUE()
    i = 0
    j = len(imagesArray)
    for im in imagesArray:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        im['brisque'] = brisq.get_score(im['frame'])
        i = i + 1
    print("\n")
    return imagesArray
    # print('Reference image: %s' % brisq.get_score(path))

def score(imgArr):
    i = 0
    j = len(imgArr)
    print()
    for im in imgArr:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        #im['brisque'] = b.test_measure_BRISQUE(im['path'])
        im['brisque'] = b.test_measure_BRISQUE(im['frame'],False)
        i = i + 1
    print("\n")
    return imgArr

def brisqueThread(images):
    with Pool(5) as p:
        ret = p.map(scoreSingle, images)
    return ret

def scoreSingle(img):
    img['brisque'] = b.test_measure_BRISQUE(img['frame'],False)
    return img

def multipleBrisque2(imagesArray):
    brisq = BRISQUE()
    i = 0
    j = len(imagesArray)
    for im in imagesArray:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        im['brisque'] = b.test_measure_BRISQUE(im['frame'],brisq._scale_feature(brisq.get_feature(im['frame'])))
        i = i + 1
    print("\n")
    return imagesArray
    # print('Reference image: %s' % brisq.get_score(path))

def cvbrisque(imgArr):
    i = 0
    j = len(imgArr)
    for im in imgArr:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        #im['brisque'] = b.test_measure_BRISQUE(im['path'])
        try:
            im['brisque'] = brisque.score(im['img'])
        except:
            im['brisque'] = 0
        print(im['image_id'] + " " + str(im['brisque']))
        i = i + 1
    print("\n")
    return imgArr
