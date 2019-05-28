#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:08:16 2019

@author: francois
"""

import os
import numpy as np # linear algebra
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

def SaveHistory(csvFilename, history):
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = csvFilename
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


###############################################################################
# Variables
###############################################################################
PATH = "/home/francois/Development/DataSet/dogs-vs-cats/train"
VALID_SPIT = 0.2
IMAGE_SIZE = 64
IMAGE_CHANNELS = 1
MAX_DATA = len(os.listdir(PATH))
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-6
###############################################################################


##############################################################################
 Processing data
##############################################################################
label_cat = []
label_dog = []
data_cat  = []
data_dog  = []
counter   = 0

for file in os.listdir(PATH):
    if counter < MAX_DATA:
        image_data = cv2.imread(os.path.join(PATH,file), cv2.IMREAD_GRAYSCALE)
        image_data = cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
        if file.startswith("cat"):
            label_cat.append(1)
            #data_cat.append(image_data[:,:,::-1]/255) # opencv read BGR plt imshow wxpect RGB we need to reverse the last channel
            data_cat.append(image_data/255) # opencv read BGR plt imshow wxpect RGB we need to reverse the last channel
        elif file.startswith("dog"):
            label_dog.append(0)
            #data_dog.append(image_data[:,:,::-1]/255) # opencv read BGR plt imshow wxpect RGB we need to reverse the last channel
            data_dog.append(image_data/255) # opencv read BGR plt imshow wxpect RGB we need to reverse the last channel
        counter += 1
        
        if counter%1000 == 0:
            print (counter," image data retreived")

data_cat = np.array(data_cat)
data_dog = np.array(data_dog)
label_cat = np.array(label_cat)
label_dog = np.array(label_dog)

print (data_cat.shape)
print (label_cat.shape)
print (data_dog.shape)
print (label_dog.shape)

train_cat_data, valid_cat_data, train_cat_label, valid_cat_label = train_test_split(
        data_cat, label_cat, test_size = VALID_SPIT, random_state = 42)
train_dog_data, valid_dog_data, train_dog_label, valid_dog_label = train_test_split(
        data_dog, label_dog, test_size = VALID_SPIT, random_state = 42)

train_data  = np.concatenate((train_cat_data, train_dog_data), axis = 0) 
valid_data  = np.concatenate((valid_cat_data, valid_dog_data), axis = 0)
train_label = np.concatenate((train_cat_label, train_dog_label), axis = 0) 
valid_label = np.concatenate((valid_cat_label, valid_dog_label), axis = 0)

del data_cat, label_cat
del data_dog, label_dog
del train_cat_data, valid_cat_data, train_cat_label, valid_cat_label
del train_dog_data, valid_dog_data, train_dog_label, valid_dog_label

train_data, train_label = shuffle(train_data, train_label, random_state=0)
valid_data, valid_label = shuffle(valid_data, valid_label, random_state=0)
train_data = train_data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
valid_data = valid_data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
###############################################################################



###############################################################################
# CNN creation
###############################################################################
# Create the MLP
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', 
                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile model using accuracy to measure model performance
optimizer = keras.optimizers.Adam(lr=LR)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()
###############################################################################



###############################################################################
# Fit the model
###############################################################################
batch_size = [32,64,128,256,512]

for bsize in batch_size:
    print (bsize)
    historyFileName = "DogsCatsClassificationKerasHistoryBatchSize" + str(bsize) + ".csv"
    print (historyFileName)

    history = model.fit(train_data,
                        train_label, 
                        epochs = EPOCHS, 
                        verbose = 2,
                        batch_size = bsize, 
                        validation_data = (valid_data, valid_label))
    
    # convert the history.history dict to a pandas DataFrame:     
    SaveHistory(historyFileName, history)
    #model.save("DogsCatsClassificationKerasModel.h5")
###############################################################################
