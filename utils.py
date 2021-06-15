import tensorflow.keras as keras
from iqa.src.utils import utils
import SlideshowMaker as sl
from os import listdir, path, remove
import nima as nima
import json
from threading import Thread
import sys
import argparse
from myresize import resize2

# Variables
#
# Paths:


original = path.abspath("./Photos/original")
outputPath = path.abspath("./output")
tecnicalChanged = path.abspath("/Photos/TecnicalChanged")

# Slideshow variables:
# fps = frames per second
# tF = transition frames - number must be an int > 0
# imgF = image frames - number must be an int > 0
# imgSec = seconds per image
# tSec = seconds per image transiction

fps = 10
imgSec = 1
tSec = 0.5
nImages = 15
models = ['mobilenet_aesthetic', 'mobilenet_technical']

def loadImages(images):
    for image in images:
        img,size = utils.load_image(image["path"], target_size=(224, 224))
        image["kerasImage"] = keras.applications.mobilenet.preprocess_input(img)
        image["size"] = size
    return images

def createSlideShow(script, path):
    imgF, tF = sl.generateFramesFromImgTime(imgSec, tSec, fps)
    images = sl.decodeHEIF(path, script)
    return imgF, tF, images


def classifyImages():
    print("NIMA aesthetic evaluation")
    result = str(nima.classify(True, myPath))
    print("Parsing NIMA results")
    result = (result[(str(result).find("step")+4):len(result)-1])
    result = list(result)
    for i in range(len(result)):
        if result[i] == "\\":
            result[i] = ""
        if result[i] == "n":
            result[i] = ""
    result = ''.join(result)
    return json.loads(result)


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def orderList(l, key, key1=False, key2=False, percent=0.5):
    if key:
        return sorted(l, key=lambda x: percent * x[key1] + x[key2] * (1-percent), reverse=True)
    else:
        return sorted(l, key=lambda x: x[key1], reverse=True)


def writeToFile(data, path):
    outfile = open(path, 'w')
    json.dump(data, outfile, indent=4)
    outfile.close()

def loadFromFile(filepath):
    with open(filepath) as json_file:
        return json.load(json_file)