from testClusters import organizeByLabelsAndObjects, testOrganizationByLabelsAndObjects
import SlideshowMaker as sl
from os import path
import os
import shutil
import nima as nima
import argparse
from myresize import resize2
from clustering import groupImages
import utils as util
import time
import visonApi
import numpy as np
import brisque2 as bq
from multiprocessing import Pool

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


def mainNimaThreads(path):
    imgF, tF, images = util.createSlideShow(False, path)
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


def _nima(images, j, string="aesthetic", model=models[0]):
    print("Get nima score")
    with Pool(5) as p:
        ret = p.map(_calculateNima, images)
    return ret
    # i = 0
    # print("Get nima score")
    # for image in images:
    #     print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
    #     image[string] = nima.getImageQuality(image["img"], model)
    #     i = i + 1
    # print("\n")
    # return images


def _orderByQuality(images, quality1, quality2=False, percent=0.5):
    images = util.orderList(images, quality2 != False,
                            quality1, quality2, percent)


def _nima(images, j, string="aesthetic", model=models[0]):
    print("Get nima score")
    with Pool(5) as p:
        ret = p.map(_calculateNima, images)
    return ret

def _imageLabelingAndClustering(images, path, j, writeToFile=False):
    print("Vision API labeling")
    start = time.time()
    objects, labels = visonApi.defineImageContent(path, images)
    if writeToFile:
        util.writeToFile(list(objects), f"{outputPath}/objects.json")
        util.writeToFile(list(labels), f"{outputPath}/labels.json")
    end = time.time()-start
    print("Image content identification per image: " + str(end/j))
    print("Clustering images based on their content")
    start = time.time()
    groups = organizeByLabelsAndObjects(objects, labels, images)
    end = time.time()-start
    print("Timr per image grouping: " + str(end/j))
    return groups


def _getTotalNumberofFrames(tF, imgF, nImages):
    return nImages * (tF+imgF)-(tF*2)


def _selectImagesToDisplay(groups, quality1, quality2, percent, ratio):
    images = list()
    for group in groups:
        tmp = round(len(groups) * ratio) + 1
        images = np.concatenate((images, group[:tmp]))
    images = _orderByQuality(images, quality1, quality2, percent)
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
    util.writeToFile([{f"{quality1}{((' '+quality2) if quality2 != False else '')}": _getQualityScore(key[quality1], key[quality2], percent), "objects": key["objects-score"], "labels": key["labels-score"], "image_id":key["image_id"],
                     "imagePath": "file://" + path + "/" + key["image_id"]}for key in images], outputPath+"/"+fileName+".json")
    print('Saving image quality eval to : ' +
          outputPath+"/"+fileName+".json")


def mainNima(model, string, string2order, path):
    print(path)
    # Get images path & generate slideshow information
    imgF, tF, images = util.createSlideShow(True, path)
    # Get images for Nima
    images = nima.loadAndPreprocessImage(images)
    images, w, h = sl.loadCV2Img(images, True, 25)
    print("Get image quality")
    i = 0
    j = len(images)
    # Run Brisque
    print("Get brisque score")
    start = time.time()
    images = bq.brisqueThread(images)
    end = time.time()-start
    print("Brisque eval per image: " + str(end/j))
    # Run Nima
    print("Get nima score")
    for image in images:
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        image[string] = nima.getImageQuality(image["img"], model)
        i = i + 1
    print("\n")
    # Order images based on aesthetics
    images = util.orderList(images, False, string2order, string)
    #print("Image clustering:")
    # kerasClustering(images,path,kerasOutput)
    print("Vision API labeling")
    start = time.time()
    objects, labels = visonApi.defineImageContent(path, images)
    util.writeToFile(list(objects), f"{outputPath}/objects,json")
    util.writeToFile(list(labels), f"{outputPath}/labels,json")
    end = time.time()-start
    print("Image content identification per image: " + str(end/j))
    print("Clustering images based on their content")
    start = time.time()
    groups = organizeByLabelsAndObjects(objects, labels, images)
    end = time.time()-start
    print("Timr per image grouping: " + str(end/j))
    # Get images for slideshow
    totalNumberOfFrames = nImages * (tF+imgF)-(tF*2)
    images, w, h = sl.loadCV2Img(images)
    images = resize2(images, w, h)
    # Generate slideshow
    sl.write_video(outputPath + "/out.mp4",
                   images[:nImages], w, h, totalNumberOfFrames, fps, tF, imgF)
    util.writeToFile([{string2order: key[string2order], "objects": key["objects-score"], "labels": key["labels-score"], "image_id":key["image_id"],
                     "imagePath": "file://" + path + "/" + key["image_id"]}for key in images], outputPath+"/"+string2order+".json")
    print('Saving image quality eval to : ' +
          outputPath+"/"+string2order+".json")


def mainClustering(path, targetdir):
    imgF, tF, images = util.createSlideShow(True, path)
    images = nima.loadAndPreprocessImage(images)
    images, w, h = sl.loadCV2Img(images, True, 25)
    groups = groupImages(images, 0.075)
    i = 1
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    for group in groups:
        if(len(group) > 1):
            print("Group " + str(i))
            i = i + 1
            for im in group:
                print(im["image_id"])
                shutil.copy(path + "/" + im["image_id"], targetdir +
                            "/" + str(i) + "_" + im["image_id"] + ".jpg")
    util.writeToFile([[{"imagePath": "file://" + path + "/" + key["image_id"]}
                     for key in group] for group in groups], outputPath+"/clustering.json")


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
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./thesis-312720-c0663e5d4e21.json"
    # Main with nima + brisque -- ordered by brisqye
    # mainNima(models[0], "aesthetic", "brisque", original)
    imgF, tF, images, w, h, j = _loadImages(original)
    images = _brisque(images, j)
    start = time.time()
    images = _nima(images, j)
    end = time.time() - start
    print("Nima eval per image: " + str(end/j))
    print("\n")
    _orderByQuality(images, "brisque", "aesthetic", 0.5)
    #groups = _imageLabelingAndClustering(images,original,j)
    groups = testOrganizationByLabelsAndObjects()
    totalNumberOfFrames = _getTotalNumberofFrames(tF, imgF, nImages)
    print(groups)
    images = _selectImagesToDisplay(
        groups, "brisque", "aesthetic", 0.5, nImages/j)
    _generateSlideShow(images, totalNumberOfFrames, tF,
                       imgF, "brisque", "aesthetic", 0.5)
    # mainNima(models[0], "aesthetic", "brisque", tecnicalChanged)
    # Main to test clustering
    # mainClustering(original,orbOutput)

    # kerasClustering(original,kerasOutput)
