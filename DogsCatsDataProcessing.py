# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:55:02 2019

@author: francois
"""
import argparse
import os
import cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

###############################################################################
# Variables
###############################################################################
PATH = "/home/francois/Development/DataSet/dogs-vs-cats/train"
VALID_SPIT = 0.2
IMAGE_SIZE = 64
IMAGE_CHANNELS = 1
MAX_DATA = 20000 #len(os.listdir(PATH))
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-6
###############################################################################


#------------------------------------------------------------------------------
# Processing data
#------------------------------------------------------------------------------
def ProcessData(dataDir, maxData):
    
    label_cat = []
    label_dog = []
    data_cat  = []
    data_dog  = []
    
    numberCats = maxData/2
    numberDogs = maxData/2
    countCats = 0
    countDogs = 0
    files = os.listdir(dataDir)
    count = 0
    
    print("numberCats: {}".format(numberCats))
    print("numberDogs: {}".format(numberDogs))
    
    while countCats < numberCats and countDogs < numberDogs:
        file = files[count]
        image_data = cv2.imread(os.path.join(PATH,file), cv2.IMREAD_GRAYSCALE)
        print(image_data.shape)
        #image_data = cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
        if file.startswith("cat") and countCats < numberCats :
            label_cat.append(1)
            #data_cat.append(image_data[:,:,::-1]/255) # opencv read BGR plt imshow wxpect RGB we need to reverse the last channel
            data_cat.append(image_data/255) # opencv read BGR plt imshow wxpect RGB we need to reverse the last channel
        elif file.startswith("dog") and countDogs < numberDogs :
            label_dog.append(0)
            #data_dog.append(image_data[:,:,::-1]/255) # opencv read BGR plt imshow wxpect RGB we need to reverse the last channel
            data_dog.append(image_data/255) # opencv read BGR plt imshow wxpect RGB we need to reverse the last channel
        count += 1
        
        if count%1000 == 0:
            print (count," image data retreived")

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
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Application
#------------------------------------------------------------------------------
def main():    
    parser = argparse.ArgumentParser(
        description = 'Preprcoss Data script',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataDir',
                        type = str,
                        help = 'directory containing data.')
    parser.add_argument('dataFile',
                        type = str,
                        help = 'numpy file containing processed data.')
    parser.add_argument('-m', '--maxData',
                        type = int,
                        help = 'Number of run on the data.')
    args = parser.parse_args()
	    
    if args.maxData == None:
        args.maxData = len(os.listdir(PATH))
    
    ProcessData(args.dataDir,args.maxData)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
#------------------------------------------------------------------------------