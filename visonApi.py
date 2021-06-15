from os import path
from google.cloud import vision

def getImageContent(path,client):
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations
    objSet = set()
    objs = []
    for o in objects:
        if o.name not in objSet:
            objs.append(f"{o.name} - {o.score}")
            objSet.add(o.name)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    lblSet = set()
    lbls = []
    for l in labels:
        if l.description not in lblSet:
            lbls.append(f"{l.description} - {l.score}")
            lblSet.add(l.description)
    return objSet, objs, lblSet, lbls
    #return list(set([o.name for o in objects])), list(set([label.description for label in labels]))

def defineImageContent(path, images):
    j = len(images)
    client = vision.ImageAnnotatorClient()
    labels= set()
    objects= set()
    for i,img in enumerate(images):
        print("(" + str(i+1) + " / " + str(j) + ")", end="\r")
        img["objects"],img["objects-score"], img["labels"], img["labels-score"] = getImageContent(path + "/" + img["image_id"],client)
        labels |= img["labels"]
        objects |= img["objects"]
        img["objects"] = list(img["objects"])
        img["labels"] = list(img["labels"])
    return objects, labels

def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations
    return list(set([o.name for o in objects]))

    '''print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))'''

def identifyObjects(path,images):
    for img in images:
        img["objects"] = localize_objects(path + "/" + img["image_id"])

def detect_labels(path):
    """Detects labels in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    return list(set([label.description for label in labels]))
    """for label in labels:
        print(label.description)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))"""

def defineLabels(path, images):
    for img in images:
        img["labels"] = detect_labels(path + "/" + img["image_id"])

# localize_objects(path.abspath("./Photos/original/img_0610.jpg"))
#localize_objects(path.abspath("./Photos/original/img_0610.jpg"))