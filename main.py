from dataPreprocessing import preprocessData

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn import random
import numpy as np
import pickle


# put path to your PetImages dir here
DataSets = "C:\\Users\\nikod\\Desktop\\Cats-Dogs-Recognition\\PetImages"
Categories = ["Dog", "Cat"]
IMG_SIZE = 50  # to reshape images since images are in different shape
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

    # 1 because it's a grayscale image, if I were working with color -> 3
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    pickleOut = open("X.pickle", "wb")
    pickle.dump(X, pickleOut)
    pickleOut.close()

    pickleOut = open("y.pickle", "wb")
    pickle.dump(y, pickleOut)
    pickleOut.close()

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=5, validation_split=0.3)
