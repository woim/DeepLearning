#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:08:16 2019

@author: francois
"""

import numpy as np # linear algebra
import argparse

#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense#, Activation

def SaveHistory(csvFilename, history):
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = csvFilename
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


#------------------------------------------------------------------------------
# Variables
#------------------------------------------------------------------------------
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-6
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# CNN creation
#------------------------------------------------------------------------------
def CreateModel(dataSize):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=dataSize))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
#------------------------------------------------------------------------------


##------------------------------------------------------------------------------
## Fit the model
##------------------------------------------------------------------------------
#batch_size = [32]#,64,128,256,512]
#
## compile model using accuracy to measure model performance
#optimizer = keras.optimizers.Adam(lr=LR)
#model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
#model.summary()
#
#for bsize in batch_size:
#    print (bsize)
#    historyFileName = "DogsCatsClassificationKerasHistoryBatchSize" + str(bsize) + ".csv"
#    print (historyFileName)
#
#    history = model.fit(train_data,
#                        train_label, 
#                        epochs = EPOCHS, 
#                        verbose = 2,
#                        batch_size = bsize, 
#                        validation_data = (valid_data, valid_label))
#    
#    # convert the history.history dict to a pandas DataFrame:     
#    SaveHistory(historyFileName, history)
#    #model.save("DogsCatsClassificationKerasModel.h5")
##------------------------------------------------------------------------------
    
    
#------------------------------------------------------------------------------
# Application
#------------------------------------------------------------------------------
def main():    
    parser = argparse.ArgumentParser(
        description = 'fit model',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('trainingDataFile',
                        type = str,
                        help = 'numpy file containing training data.')
    parser.add_argument('-v', '--validationDataFile',
                        type = str,
                        nargs = '?',
                        help = 'numpy file containing validation data.')    
    args = parser.parse_args()
    
    # Get data shape
    trainingData = np.load(args.trainingDataFile)
    dataSize = trainingData.item().get('data')[0].shape
    
    # Create the model
    CreateModel(dataSize)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
#------------------------------------------------------------------------------
