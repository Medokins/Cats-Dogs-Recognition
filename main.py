import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from dataPreprocessing import preprocessData

DataSets = "C:\\Users\\medok\\OneDrive\\Desktop\\Cats-Dog\\PetImages"
Categoris = ["Dog", "Cat"]

IMG_SIZE = 128 #to reshape images since images are in different shape

for category in Categoris:
    path = os.path.join(DataSets, category)
    for img in os.listdir(path):
        imgArray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #grey because I think color doesnt matters that much in this classifier
        break
    break

resizedImgArray = cv2.resize(imgArray, (IMG_SIZE, IMG_SIZE))

plt.imshow(resizedImgArray, cmap = "gray")
plt.show()