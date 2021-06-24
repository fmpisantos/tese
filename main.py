from algoritmos.testClusters import organizeByLabelsAndObjects, testOrganizationByLabelsAndObjects
import algoritmos.SlideshowMaker as sl
from os import path
import os
import algoritmos.nima as nima
import argparse
from algoritmos.myresize import resize2
import algoritmos.utils as util
import time
import algoritmos.visonApi as visonApi
import numpy as np
import algoritmos.brisque2 as bq
from multiprocessing import Pool
import warnings
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        from datetime import datetime

        now = datetime.now() # current date and time
        self.log = open(f"./logs/LOG_{now.strftime('%m_%d_%Y_%H_%M_%S')}.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()
# Variables
# kerasClustering
# Paths:

original = path.abspath("./Photos/original")
outputPath = path.abspath("./output")
tecnicalChanged = path.abspath("./Photos/TecnicalChanged")
kerasOutput = path.abspath("./outputPath/")
orbOutput = path.abspath("./orboutput/")
brisqueTest = path.abspath("./Photos/TecnicalChangedByTheme")

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


def _loadImages(path):
    # Get images path & generate slideshow information
    imgF, tF, images = util.createSlideShow(True, path)
    # Get images
    images = nima.loadAndPreprocessImage(images)
    images, w, h = sl.loadCV2Img(images, True, 25)
    return imgF, tF, images, w, h, len(images)


def _brisque(images, j):
    print("Get brisque score")
    start = time.time()
    images = bq.brisqueThread(images)
    end = time.time()-start
    print("Brisque eval per image: " + str(end/j))
    print("\n")
    return images


def _calculateNima(image, string="aesthetic", model=models[0]):
    image[string] = 100 - (10 * nima.getImageQuality(image["img"], model))
    return image


def _nima(images):
    print("Get nima score")
    with Pool(6) as p:
        ret = p.map(_calculateNima, images)
    return ret


def _orderByQuality(images, quality1, quality2=False, percent=0.5):
    return util.orderList(images, quality2 != False,
                          quality1, quality2, percent)


def _nima(images, j, string="aesthetic", model=models[0]):
    print("Get nima score")
    with Pool(6) as p:
        ret = p.map(_calculateNima, images)
    return ret


def _imageLabelingAndClustering(images, path, j, writeToFile=False, getLabels=True, getObjects=True):
    print("Vision API labeling")
    start = time.time()
    objects, labels, labeledImages = visonApi._defineImageContent(
        path, images, getLabels=getLabels, getObjects=getObjects)
    if writeToFile:
        util.writeToFile(list(objects), f"{outputPath}/objects.json")
        util.writeToFile(list(labels), f"{outputPath}/labels.json")
    end = time.time()-start
    print("Image content identification per image: " + str(end/j))
    print("Clustering images based on their content")
    start = time.time()
    groups = organizeByLabelsAndObjects(objects, labels, labeledImages)
    end = time.time()-start
    print("Time per image grouping: " + str(end/j))
    return groups


def _getTotalNumberofFrames(tF, imgF, nImages):
    return nImages * (tF+imgF)-(tF*2)


def _selectImagesToDisplay(groups, quality1, quality2, percent, ratio):
    images = list()
    groupQuality = list()
    for group in groups:
        tmp = round(len(groups) * ratio) + 1
        images = np.concatenate((images, group[:tmp]))
        if quality2 == "aesthetic" or quality1 == "aesthetic":
            groupQuality.append({'group':group[0]['group'], 'quality': sum(c['aesthetic'] for c in group[:tmp])/len(group[:tmp])})
    images = _orderByQuality(images, quality1, quality2, percent)
    if quality2 == "aesthetic" or quality1 == "aesthetic":
        groupQuality = sorted(groupQuality, key=lambda x: x['quality'], reverse=True)
        returnList = list()
        for group in groupQuality:
            for img in images:
                if img["group"] == group["group"]:
                    returnList.append(img)
        return returnList
    return images


def _getQualityScore(q1, q2, p):
    if q2:
        return(p * q1 + q2 * (1-p))
    else:
        return q1


def _generateSlideShow(images, totalNumberOfFrames, tF, imgF, quality1, quality2=False, percent=0.5, fileName="output"):
    # Get images for slideshow
    images, w, h = sl.loadCV2Img(images)
    images = resize2(images, w, h)
    # Generate slideshow
    sl.write_video(outputPath + "/out.mp4",
                   images[:nImages], w, h, totalNumberOfFrames, fps, tF, imgF)
    util.writeToFile([{f"{quality1}{((' '+quality2) if quality2 != False else '')}": _getQualityScore(key[quality1], key[quality2], percent), "objects": key["objects-score"], "labels": key["labels-score"], "features": key["features"].tolist(), "image_id":key["image_id"],
                     "imagePath": "file://" + key["path"]}for key in images], outputPath+"/"+fileName+".json")
    print('Saving image quality eval to : ' +
          outputPath+"/"+fileName+".json")
    print('Saving .mp4 slideshow to : ' +
        outputPath + "/out.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage=argparse.SUPPRESS, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-fps', '--fps', dest='fps', default=fps,
                        help='Frames per sec.', type=int,  required=False)
    parser.add_argument('-is', '--imgSec', dest='imgSec', default=imgSec,
                        help='Seconds per image.', type=float, required=False)
    parser.add_argument('-ts', '--tSec', dest='tSec', default=tSec,
                        help='Seconds per transiction.', type=float, required=False)
    parser.add_argument('-ni', '--nImage', dest='nImages', default=nImages,
                        help='Number of images to show.', type=int, required=False)
    parser.add_argument('-a', '--alg', action='append',
                        help='''Select the algoritms to use from the following list (note: this flag can be omitted to use the recomended algoritms):
    "brisque" or "b" for tecnical photo assessment
    "nima" or "n" for aesthetic photo assessment
    "labels" or "l" for image label identification
    "objects" or "o" for objects identification
    "slideshow" or "s" to create a slideshow''', required=False)
    parser.add_argument('-p', '--path', dest='path', default=original,
                        help='Path to folder holding the photos.', type=str, required=False)
    parser.print_help()
    args = parser.parse_args()
    fps = args.fps
    imgSec = args.imgSec
    tSec = args.tSec
    nImages = args.nImages
    algs = args.alg
    _path = args.path
    if(algs == None):
        algs = list()
    if len(algs) > 0:
        algs = algs[0].strip().split(" ")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./thesis-312720-c0663e5d4e21.json"
    runNima = "nima" in algs or "n" in algs or len(algs) == 0
    runBrisque = "brisque" in algs or "b" in algs or len(algs) == 0
    runLabels = "labels" in algs or "l" in algs or len(algs) == 0
    runObjects = "objects" in algs or "o" in algs or len(algs) == 0
    runSlideShow = "slideshow" in algs or "s" in algs or len(algs) == 0
    imgF, tF, images, w, h, j = _loadImages(_path)
    if runBrisque:
        images = _brisque(images, j)
    if runNima:
        start = time.time()
        images = _nima(images, j)
        end = time.time() - start
        print("Nima eval per image: " + str(end/j))
        print("\n")
    if runNima:
        if runBrisque:
            _orderByQuality(images, "brisque", "aesthetic", 0.5)
        else:
            _orderByQuality(images, "aesthetic")
    elif runBrisque:
        _orderByQuality(images, "brisque")
    if runObjects or runLabels:
        groups = _imageLabelingAndClustering(
            images, _path, j, getLabels=runLabels, getObjects=runObjects)
    # groups = testOrganizationByLabelsAndObjects()
    totalNumberOfFrames = _getTotalNumberofFrames(tF, imgF, nImages)
    if runNima:
        if runBrisque:
            images = _selectImagesToDisplay(
                groups, "brisque", "aesthetic", 0.5, nImages/j)
        else:
            images = _selectImagesToDisplay(
                groups, "aesthetic", False, 0.5, nImages/j)
    elif runBrisque:
        images = _selectImagesToDisplay(
            groups, "brisque", False, 0.5, nImages/j)
    # images = _selectImagesToDisplay(
    #     groups, "brisque", "aesthetic", 0.5, nImages/j)
    if runSlideShow:
        _generateSlideShow(images, totalNumberOfFrames, tF,
                           imgF, "brisque", "aesthetic", 0.5)
