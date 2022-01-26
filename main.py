from dataPreprocessing import preprocessData
import numpy as np
from sklearn import random
import pickle

DataSets = "C:\\Users\\medok\\OneDrive\\Desktop\\Cats-Dog\\PetImages"
Categories = ["Dog", "Cat"]
IMG_SIZE = 128 #to reshape images since images are in different shape

overRidePickle = False

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))






if overRidePickle:
    trainingData = preprocessData(DataSets, Categories, IMG_SIZE)
    random.shuffle(trainingData)

    X = []
    y = []

    for features, label in trainingData:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 because it's a grayscale image, if I were working with color -> 3l

    pickleOut = open("X.pickle", "wb")
    pickle.dump(X, pickleOut)
    pickleOut.close()

    pickleOut = open("y.pickle", "wb")
    pickle.dump(y, pickleOut)
    pickleOut.close()