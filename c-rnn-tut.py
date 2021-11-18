## -*- coding: utf-8 -*-
#"""
#Created on Wed Sep 23 00:42:54 2020
#
#@author: jonsnow
#"""
#
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline
#import cv2
#
## This Python 3 environment comes with many helpful analytics libraries installed
## It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
## For example, here's several helpful packages to load in 
#
#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#
## Input data files are available in the "../input/" directory.
## For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#
#import os
#
##path = "../datasets/Kaggle/Face Images with Marked Landmark Points/"
#path = "./Face Images with Marked Landmark Points/"
##path = "./Landmarks"
##for dirname, _, filenames in os.walk(path):
#for filename in os.listdir(path): 
#    print(os.path.join(path, filename))
#
## load the dataset
#db_face_images = np.load(path + 'face_images.npz')['face_images']
#print(db_face_images.shape)
#df_facial_keypoints = pd.read_csv(path + 'facial_keypoints.csv')
#pd.set_option('display.max_columns', None)
#
##visualising the dataframe
#df_facial_keypoints.head()


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries
import json
import codecs
import requests
import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
from io import BytesIO
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
dataset_path = "./face_detection.json"
# Any results you write to the current directory are saved as output.

# get links and stuff from json
jsonData = []
JSONPATH = dataset_path
with codecs.open(JSONPATH, 'rU', 'utf-8') as js:
    for line in js:
        jsonData.append(json.loads(line))

print(f"{len(jsonData)} image found!")

print("Sample row:")
jsonData[0]

# load images from url and save into images
images = []
for data in tqdm(jsonData):
    response = requests.get(data['content'])
    img = np.asarray(Image.open(BytesIO(response.content)))
    images.append([img, data["annotation"]])

import time
count = 1
totalfaces = 0
start = time.time()
for image in images:
    img = image[0]
    metadata = image[1]
    for data in metadata:
        height = data['imageHeight']
        width = data['imageWidth']
        points = data['points']
        if 'Face' in data['label']:
            x1 = round(width*points[0]['x'])
            y1 = round(height*points[0]['y'])
            x2 = round(width*points[1]['x'])
            y2 = round(height*points[1]['y'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            totalfaces += 1
    cv2.imwrite('./face-detection-images/face_image_{}.jpg'.format(count),img)
    count += 1
    
end = time.time()
print("Total test images with faces : {}".format(len(images)))
print("Sucessfully tested {} images".format(count-1))
print("Execution time in seconds {}".format(end-start))
print("Total Faces Detected {}".format(totalfaces))