# Photo Sortilng Utility
This is my implementation of an open source tool to help manage a large amount of folders.  

## Incentive

The purpose is to help organize large amount of unlabeled pictures in folders,  
As it can be very challanging trying to find a specific picture from an event 3 years ago inside a folder containing a few thousand of pictures, if we didn't organize the pictures in advance.  
Example of what the folders may contain:
* All the pictures taken by your phone
* All the pictures received via Whatsapp
* Any similar kind of set of pictures

## Approaches for models and feature engineering

### Feature Engineering:  
In the following project I try a few approaches to tackle the task:  
* Running an Object Decetion on every picture and set detected objects as features.  
  The OD was done by ResNet.
* Extracting metadata from each picture
* Run face detection on every picture, and set the detected faces as features based on the faces distribution

### Models:
**Unsupervised Learning**:  
* Use all the features gathered and just run a clustering model: K-Means

**Supervised Learning**:  
I labeled around 1000 of my pictures in order to divide pictures into specific categories,  
You'll find the following approaches to the models here (multi class classification):
* Take the features we already extracted previously and feed them to the following models:
  * CatBoost
  * Deep Neural Network
  * RandomForest
* Training a convolutional neural network based on classical LeNet architecture using raw pictures as the features
* Trasfer learning on ResNet network, training on the raw pictures as the features