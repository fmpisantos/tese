import SlideshowMaker as sl
from os import listdir, path, remove
import nima as nima
import json
from threading import Thread
import sys
import argparse

def createSlideShow():
    imgF,tF = sl.generateFramesFromImgTime(imgSec,tSec,fps)
    images = sl.decodeHEIF(myPath)
    return imgF,tF,images


def classifyImages():
    print("NIMA aesthetic evaluation")
    result = str(nima.classify(True,myPath))
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
    def __init__(self, group=None, target=None, name=None,args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,**self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def orderList(l,key):
    return sorted(l, key=lambda x : x[key], reverse=True)

def writeToFile(data,path):
    outfile = open(path, 'w')
    json.dump(data, outfile, indent=4)
    outfile.close()

# Variables
#
# Paths:

myPath = path.abspath("./Photos/original")
outputPath = path.abspath("./output")

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


# def write_video(file_path,frames, w, h, totalNumberOfFrames, fps, tF, imgF):

def main():
    imgF,tF,images = createSlideShow()
    p1 = ThreadWithReturnValue(target=classifyImages)
    p1.start()
    p2 = ThreadWithReturnValue(target=sl.loadImagesToSys, args=(outputPath + "/out.mp4", images, fps, tF, imgF, nImages))
    p2.start()
    nimaResult = p1.join()
    frames,w,h,totalNumberOfFrames = p2.join()
    for x in nimaResult:
        for y in frames:
            if len(y["image_id"].split(x["image_id"]))>1:
                y["aesthetic"] = x["mea_score_predictio"]
                break
    frames = orderList(frames,"aesthetic")
    print("FPS {%d} Tempo de imagem no ecrã {%s} Numero de frames por imagem (static){%d} Numero de frames por transição {%d x 2} Numero total de imagens {%d}" % (fps,str((tF*2+imgF)/fps),imgF,tF,nImages))

    sl.write_video(outputPath + "/out.mp4", frames[:nImages], w, h, totalNumberOfFrames, fps, tF, imgF)

    # To check best images
    writeToFile( [{"aesthetic":key["aesthetic"],"image_id":key["image_id"]} for key in frames],outputPath+"/aesthetic.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
    parser.add_argument('-fps', '--fps', dest='fps', default=fps, help='Frames per sec.', type=int,  required=False)
    parser.add_argument('-is', '--imgSec',dest='imgSec', default=imgSec, help='Seconds per image.', type=int, required=False)
    parser.add_argument('-ts', '--tSec', dest='tSec', default=tSec, help='Seconds per transiction.', type=int, required=False)
    parser.add_argument('-ni', '--nImage', dest='nImages', default=nImages  , help='Number of images to show.', type=int, required=False)
    parser.print_help()
    args = parser.parse_args()
    fps = args.fps
    imgSec = args.imgSec
    tSec = args.tSec
    nImages = args.nImages
    main()
    # nima.getImageQuality(myPath+"/img (25).jpg",models[0])

