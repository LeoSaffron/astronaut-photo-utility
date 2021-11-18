# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:48:32 2020

@author: jonsnow
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 02:13:15 2020

@author: jonsnow
"""

import cv2
import face_recognition
import dlib
import matplotlib.pyplot as plt

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image 
import skimage
from skimage.io import imread

pic_limit = 10000
def load_images_from_folder(folder): 
    images = [] 
    pic_count = 0
    for filename in os.listdir(folder): 
        if(filename[-4:] == '.jpg'):
            img = skimage.io.imread(folder + filename) 
            img = cv2.imread(folder + filename) 
            images.append(img) 
            pic_count += 1
            if (pic_count > pic_limit):
                break
    return images 
images_path = "test_pictures/"

images = load_images_from_folder(images_path)

faces = []
image_count = 0
for img in images:
    faces.append(face_recognition.face_locations(img,model = "cnn"))
    image_count += 1
    print("faces detected on {} pictures".format(image_count))
    if (image_count % 5 == 0):
        print("faces detected on {} pictures".format(image_count))

for i in range(len(images)):
    index = 0
    temp_img = images[i].copy()
    for detected_face in faces[i]:
        y0, x1, y1, x0 = detected_face
        print(x0, x1, y0, y1)
    #    cv2.imshow("Face {}".format(index), image_to_detect[y0:y1, x0:x1])
        img = cv2.rectangle(temp_img, (x0,y0), (x1,y1), (0,0,255), 3)
#        faces.append(images[i][y0:y1, x0:x1])
        index += 1
    plt.imshow(img)
    plt.show()
        
    