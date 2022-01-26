import os
import cv2

def preprocessData(data, Categories, imgSize):

    trainingData = []

    for category in Categories:
        path = os.path.join(data, category)
        classNumber = Categories.index(category)
        for img in os.listdir(path):
            try:
                imgArray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #grey because I think color doesnt matters that much in this classifier
                resizedArray = cv2.resize(imgArray, (imgSize, imgSize))
                trainingData.append([resizedArray, classNumber])
            except: #some of the images are broken, but it's only a few so I don't bother fixing them
                pass

    return trainingData