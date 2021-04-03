import tensorflow.keras as keras
from iqa.src.utils import utils

def loadImages(images):
    for image in images:
        img,size = utils.load_image(image["path"], target_size=(224, 224))
        image["kerasImage"] = keras.applications.mobilenet.preprocess_input(img)
        image["size"] = size
    return images