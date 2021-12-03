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

from mtcnn.mtcnn import MTCNN
import mtcnn
import face_recognition as fr


detection_mode = "mtcnn"
mode_face_recognition = "face_recognition_package_1_3_0"


detector_mtcnn = MTCNN()


pic_limit = 10000
def load_images_from_folder(folder): 
    images = [] 
    pic_count = 0
    for filename in os.listdir(folder): 
        if(filename[-4:] == '.jpg'):
            img = skimage.io.imread(folder + filename) 
            img = cv2.imread(folder + filename) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img) 
            pic_count += 1
            if (pic_count > pic_limit):
                break
    return images 
images_path = "test_pictures/"

images = load_images_from_folder(images_path)


path_known_faces_images = "./faces_known_test/"
known_names = []
known_name_encodings = []
filenames_images = os.listdir(path_known_faces_images)

for _ in filenames_images:
    image_loaded = fr.load_image_file(path_known_faces_images + _)
    image_path = path_known_faces_images + _
    encoding = fr.face_encodings(image_loaded)[0]
    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())


#faces = []
#image_count = 0
#for img in images:
#    faces.append(face_recognition.face_locations(img,model = "cnn"))
#    image_count += 1
#    print("faces detected on {} pictures".format(image_count))
#    if (image_count % 5 == 0):
#        print("faces detected on {} pictures".format(image_count))
#
#for i in range(len(images)):
#    index = 0
#    temp_img = images[i].copy()
#    for detected_face in faces[i]:
#        y0, x1, y1, x0 = detected_face
#        print(x0, x1, y0, y1)
#        img = cv2.rectangle(temp_img, (x0,y0), (x1,y1), (0,0,255), 3)
##        faces.append(images[i][y0:y1, x0:x1])
#        index += 1
#    plt.imshow(img)
#    plt.show()


def detect_faces_on_single_image_with_mtcnn(image):
    # confirm mtcnn was installed correctly
    # print version
    
    pixels = np.asarray(image)
    
    # create the detector, using default weights
    # detect faces in the image
    detection_results = detector_mtcnn.detect_faces(pixels)
    
    # extract the bounding box from the first face
    face_coordinates = []
    for detection_result_item in detection_results:
            
        x1, y1, width, height = detection_result_item['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face_coordinates.append([x1, y1, x2, y2])
    
    # extract the face
#    face = pixels[y1:y2, x1:x2]
    return face_coordinates
#    face_coordinates = x1, y1, x2, y2
#    return face_coordinates

def detect_faces_on_single_image(image):
    if (detection_mode == "mtcnn"):
#        pixels = np.asarray(image)
        detection_results = detect_faces_on_single_image_with_mtcnn(image)
        return detection_results

def image_face_recognize_with_package_face_recognition(image):
    face_encoding = None
    name = ""
    try:
        face_encoding = fr.face_encodings(image)[0]
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)
        if matches[best_match]:
            name = known_names[best_match]
    except:
        pass
    return name

def image_face_recognize(image):
    if mode_face_recognition == "face_recognition_package_1_3_0":
        return image_face_recognize_with_package_face_recognition(image)
    return "r"

def scan_faces_on_single_image(image):
    detection_results = detect_faces_on_single_image(image)
    faces = []
    pixels = np.asarray(image)
    for face in detection_results:
        x1, y1, x2, y2 = face
        faces.append(pixels[y1:y2, x1:x2])
    people_recognized = []
    for face in faces:
        result_face_recognition = image_face_recognize(face)
        people_recognized.append(result_face_recognition)
    return people_recognized

def detect_faces_on_image_list(image_list):
    pass

def run_face_recognition_predetected_on_single_image(image, faces_coordinates_list):
    pass

def run_face_recognition_predetected_on_image_list(image_list, faces_coordinates_list_list):
    pass

def get_faces_list_by_recognizing_faces(images):
    list_faces_recignized = []
    for image in images:
        faces_detected_names = scan_faces_on_single_image(image)
        list_faces_recignized.append(faces_detected_names)
    return list_faces_recignized

def run_objects_detection_on_single_image(image):
    pass

def run_objects_detection_on_image_list(image_list):
    pass





















