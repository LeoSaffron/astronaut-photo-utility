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
try:
    from keras.utils import plot_model
except:
    from keras.utils.vis_utils import plot_model
from keras import backend as K

import numpy as np
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
import tensorflow as tf
import pathlib
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


detection_mode = "mtcnn"
mode_face_recognition = "face_recognition_package_1_3_0"


detector_mtcnn = MTCNN()


pic_limit = 200

def get_images_path_from_folder(folder): 
    path_list = [] 
    pic_count = 0
    for filename in os.listdir(folder): 
        if(filename[-4:] == '.jpg'):
            path_list.append(folder + filename) 
            pic_count += 1
            if (pic_count > pic_limit):
                break
    return path_list 

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


def get_ssd_resnet_model_from_net():
    url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz'
    
    PATH_TO_MODEL_DIR = tf.keras.utils.get_file(
        fname='ssd_resnet101_v1_fpn_640x640_coco17_tpu-8',
        origin=url,
        untar=True)
    
    # PATH_TO_MODEL_DIR
    
    # url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'
    
    # PATH_TO_LABELS = tf.keras.utils.get_file(
    #     fname='mscoco_label_map.pbtxt',
    #     origin=url,
    #     untar=False)
    
    
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    
    print('Loading model...', end='')
    start_time = time.time()
    
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn

def get_ssd_resnet_path_to_labels():
    url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'
    return tf.keras.utils.get_file(
        fname='mscoco_label_map.pbtxt',
        origin=url,
        untar=False)




def detect_objects_ssd_resnet_by_image_numpy(image, model, category_index, threshold):
    
    image_np = image
    
    input_tensor = tf.convert_to_tensor(image_np)
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Do the detection
    detections = model(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    # show classes
    # unique_classes = set(detections['detection_classes'])
    # print("Classes found:")
    # for c in unique_classes:
    #     print(category_index[c]['name'])
    
    image_np_with_detections = image_np.copy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(image_np_with_detections)
    print('Done')
    plt.show()

    result_detections_int = []
    result_detections_str = []
    for i in range(len(detections['detection_scores'])):
        if detections['detection_scores'][i] > threshold:
            result_detections_int.append(detections['detection_classes'][i])
            result_detections_str.append(category_index[detections['detection_classes'][i]]['name'])
    return result_detections_int, result_detections_str

def detect_objects_ssd_resnet_by_image_path(path, model, category_index, threshold):
    image_np = np.array(Image.open(path))
    return detect_objects_ssd_resnet_by_image_numpy(image, model, category_index, threshold)



def run_objects_detection_on_single_image_resnet(image, threshold=0.3):
    global model_ssd_resnet, category_index_ssd_resnet
    return detect_objects_ssd_resnet_by_image_numpy(image,
                model_ssd_resnet, category_index_ssd_resnet, threshold=threshold)

def run_objects_detection_on_single_image(image, threshold=0.3):
    return run_objects_detection_on_single_image_resnet(image, threshold=threshold)

def run_objects_detection_on_image_list(image_list, threshold=0.3):
    results = []
    for image in image_list:
        results.append(run_objects_detection_on_single_image(image, threshold=0.3))
    return results

def run_clustering_on_detected_faces(faces):
    pass

def run_clustering_on_detected_objects(faces):
    pass


# detect_fn = get_ssd_resnet_model_from_net()

model_ssd_resnet = get_ssd_resnet_model_from_net()

category_index_ssd_resnet = label_map_util.create_category_index_from_labelmap(
    get_ssd_resnet_path_to_labels(),
    use_display_name=True)



def get_encoding_of_detected_objects_for_single_result(result_item):
    result_encoded = [0] * 1000
    for detected_object in result_item[0]:
        result_encoded[detected_object] += 1
        return result_encoded

def get_encoding_of_detected_objects(results_detected_objects):
    result = []
    for result_item in results_detected_objects:
        result.append(get_encoding_of_detected_objects_for_single_result(result_item))
    return result


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, n_init=20, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(np.asarray(a))



mymodel = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)


from skimage.transform import resize




images_test = []
for image in images:
    images_test.append(resize(image, (224, 224)))


# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# model = ResNet50(weights='imagenet')



# img = image.load_img(images_path + os.listdir(images_path)[0], target_size=(224, 224))


# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=30)[0])

# from tensorflow.keras.applications.vgg16 import VGG16

# model_vgg16 = VGG16(weights='imagenet', include_top=False)


# features_vgg16 = model_vgg16.predict(x)
# print('Predicted:', decode_predictions(preds_vgg16, top=30)[0])



























