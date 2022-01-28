from dataPreprocessing import *

import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn import random
import numpy as np
import pickle
import time

NAME = f"Cats-Dogs-Recognition-cnn-64x2{int(time.time())}"
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# put path to your PetImages dir here
DataSets = "C:\\Users\\nikod\\Desktop\\Cats-Dogs-Recognition\\PetImages"
Categories = ["Dog", "Cat"]
IMG_SIZE = 50  # to reshape images since images are in different shape
overRidePickle = False
retrainModel = False

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

if not overRidePickle:
    X = pickle.load(open("X.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))

if retrainModel:

    X = X/255.0

    dense_layers = [1]  # originally 0, 1, 2
    layer_sizes = [64]  # originally 32, 54, 128
    conv_layers = [3]  # originally 1, 2, 3

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
                    conv_layer, layer_size, dense_layer, int(time.time()))

                model = Sequential()

                model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'],
                              )

                model.fit(X, y,
                          batch_size=32,
                          epochs=10,
                          validation_split=0.3,
                          callbacks=[tensorboard])

    model.save("64x3-CNN.model")


model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('testDoggo.jpg')])
print(Categories[int(prediction[0][0])])
