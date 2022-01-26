from dataPreprocessing import preprocessData
import numpy as np
from sklearn import random

DataSets = "C:\\Users\\medok\\OneDrive\\Desktop\\Cats-Dog\\PetImages"
Categories = ["Dog", "Cat"]
IMG_SIZE = 128 #to reshape images since images are in different shape

trainingData = preprocessData(DataSets, Categories, IMG_SIZE)
random.shuffle(trainingData)

X = []
y = []

for features, label in trainingData:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 because it's a grayscale image, if I were working with color -> 3l