import SlideshowMaker as sl
from os import listdir, path, remove
import nima as nima
import json
from threading import Thread
import sys
import argparse
from myresize import resize2
from clustering import groupImages, kerasClustering
import utils as util
import time

import brisque2 as bq

# Variables
#kerasClustering
# Paths:


original = path.abspath("./Photos/original")
outputPath = path.abspath("./output")
tecnicalChanged = path.abspath("./Photos/TecnicalChanged")
kerasOutput = path.abspath("./outputPath/")

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

def mainNimaThreads(path):
    imgF, tF, images = util.createSlideShow(False,path)
    p1 = ThreadWithReturnValue(target=classifyImages)
    p1.start()
    p2 = ThreadWithReturnValue(target=sl.loadImagesToSys, args=(
        outputPath + "/out.mp4", images, fps, tF, imgF, nImages))
    p2.start()
    nimaResult = p1.join()
    frames, w, h, totalNumberOfFrames = p2.join()
    for x in nimaResult:
        for y in frames:
            if len(y["image_id"].split(x["image_id"])) > 1:
                y["aesthetic"] = x["mea_score_predictio"]
                break
    frames = util.orderList(frames, "aesthetic")
    print("FPS {%d} Tempo de imagem no ecrã {%s} Numero de frames por imagem (static){%d} Numero de frames por transição {%d x 2} Numero total de imagens {%d}" % (
        fps, str((tF*2+imgF)/fps), imgF, tF, nImages))

    sl.write_video(outputPath + "/out.mp4",
                   frames[:nImages], w, h, totalNumberOfFrames, fps, tF, imgF)

    # To check best images
    util.writeToFile([{"aesthetic": key["aesthetic"], "image_id":key["image_id"]}
                for key in frames], outputPath+"/aesthetic.json")


def mainNima(model, string, string2order, path):
    # Get images path & generate slideshow information
    imgF, tF, images = util.createSlideShow(True,path)
    # Get images for Nima
    images = nima.loadAndPreprocessImage(images)
    images, w, h = sl.loadCV2Img(images,True,25)
    # Run Nima
    print("Get image quality")
    i = 0
    j = len(images)
    print("Get brisque score")
    start = time.time()
    images = bq.brisqueThread(images)
    # bq.score(images)
    # bq.multipleBrisque(images)
    # bq.cvbrisque(images)
    end = time.time()-start
    print("Brisque eval per image: " + str(end/j))
    #bq.multipleBrisque(images)
    print("Get nima score")
    for image in images:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        image[string] = nima.getImageQuality(image["img"], model)
        i = i + 1
    print("\n")
    # Order images based on aesthetics
    images = util.orderList(images, string2order)
    # Get images for slideshow
    totalNumberOfFrames = nImages * (tF+imgF)-(tF*2)
    images, w, h = sl.loadCV2Img(images)
    images = resize2(images, w, h)
    # Generate slideshow
    sl.write_video(outputPath + "/out.mp4",
                   images[:nImages], w, h, totalNumberOfFrames, fps, tF, imgF)
    util.writeToFile([{string2order: key[string2order], "image_id":key["image_id"], "imagePath": "file://" + path + "/" + key["image_id"]}for key in images], outputPath+"/"+string2order+".json")


def mainClustering(path):
    imgF, tF, images = util.createSlideShow(True,path)
    images = nima.loadAndPreprocessImage(images)
    images, w, h = sl.loadCV2Img(images,True,25)
    groups = groupImages(images, 0.1)
    i = 1
    for group in groups:
        if(len(group) > 1):
            print("Group " + str(i))
            i = i + 1
            for im in group:
                print(im["image_id"])

    util.writeToFile([[{"imagePath": "file://" + path + "/" + key["image_id"]}for key in group] for group in groups], outputPath+"/clustering.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
    parser.add_argument('-fps', '--fps', dest='fps', default=fps,
                        help='Frames per sec.', type=int,  required=False)
    parser.add_argument('-is', '--imgSec', dest='imgSec', default=imgSec,
                        help='Seconds per image.', type=int, required=False)
    parser.add_argument('-ts', '--tSec', dest='tSec', default=tSec,
                        help='Seconds per transiction.', type=int, required=False)
    parser.add_argument('-ni', '--nImage', dest='nImages', default=nImages,
                        help='Number of images to show.', type=int, required=False)
    parser.print_help()
    args = parser.parse_args()
    fps = args.fps
    imgSec = args.imgSec
    tSec = args.tSec
    nImages = args.nImages
    # Main with nima + brisque -- ordered by brisqye
    # mainNima(models[1], "tecnical", "brisque", tecnicalChanged)
    # Main to test clustering 
    # mainClustering(original)

    kerasClustering(original,kerasOutput)
