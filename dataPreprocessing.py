import os
import cv2


def preprocessData(data, Categories, imgSize):

    trainingData = []

    for category in Categories:
        path = os.path.join(data, category)
        classNumber = Categories.index(category)
        for img in os.listdir(path):
            try:
                # grey because I think color doesnt matters that much in this classifier
                imgArray = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                resizedArray = cv2.resize(imgArray, (imgSize, imgSize))
                trainingData.append([resizedArray, classNumber])
            except:  # some of the images are broken, but it's only a few so I don't bother fixing them
                pass

    return trainingData


def prepare(filepath):
    IMG_SIZE = 50
    imgArray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    newArray = cv2.resize(imgArray, (IMG_SIZE, IMG_SIZE))
    return newArray.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
