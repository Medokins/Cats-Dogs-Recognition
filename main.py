from dataPreprocessing import preprocessData
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
from sklearn import random
import pickle

DataSets = "C:\\Users\\medok\\OneDrive\\Desktop\\Cats-Dog\\PetImages" #put path to your PetImages dir here
Categories = ["Dog", "Cat"]
IMG_SIZE = 128 #to reshape images since images are in different shape
overRidePickle = False

if not overRidePickle:
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

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 because it's a grayscale image, if I were working with color -> 3
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #1 because it's a grayscale image, if I were working with color -> 3l

    pickleOut = open("X.pickle", "wb")
    pickle.dump(X, pickleOut)
    pickleOut.close()

    pickleOut = open("y.pickle", "wb")
    pickle.dump(y, pickleOut)
    pickleOut.close()

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))  #(3,3) is the window size that my Conv will work with
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])

model.fit(X,y, batch_size = 32, epochs = 10, validation_split=0.1)
