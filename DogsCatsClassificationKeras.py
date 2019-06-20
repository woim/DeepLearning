#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:08:16 2019

@author: francois
"""

import numpy as np # linear algebra
import argparse
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense#, Activation


#------------------------------------------------------------------------------
# Variables
#------------------------------------------------------------------------------
def SaveHistory(csvFilename, history):
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = csvFilename
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
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
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    return model
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Fit the model
#------------------------------------------------------------------------------
def FitModel(model, training, learningRate, epochs, batchSize, validation = None):
    
    # Optimizer
    optimizer = keras.optimizers.Adam(lr=learningRate)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(training['data'],
                        training['label'], 
                        epochs = epochs, 
                        verbose = 2,
                        batch_size = batchSize, 
                        validation_data = validation)
    
    return history
#------------------------------------------------------------------------------
    
    
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
    parser.add_argument('-m', '--modelFile',
                        type = str,
                        nargs = '?',
                        help = 'file to store the model.')
    parser.add_argument('-t', '--trainingHistoryFile',
                        type = str,
                        nargs = '?',
                        help = 'file to store the training history.')
    parser.add_argument('-e', '--epochs',
                        type = int,
                        default = 200,
                        help = 'Number of epochs for trainig data.')
    parser.add_argument('-lr', '--learningRate',
                        type = float,
                        default = 1e-6,
                        help = 'Learning rate for training.')
    parser.add_argument('-b', '--batchSize',
                        type = int,
                        default = 64,
                        help = 'Learning rate for training.')
    args = parser.parse_args()
    
    # Get training data shape
    trainingData = np.load(args.trainingDataFile)
    training = {
            'data': trainingData.item().get('data') ,
            'label': trainingData.item().get('label')}
    dataSize = training['data'][0].shape
    
    # Get validationdata
    validation = None
    if args.validationDataFile is not None:
        data = np.load(args.validationDataFile)
        validation = (data.item().get('data'), data.item().get('label'))
        
    # Create the model
    model = CreateModel(dataSize)
    
    # Fit model
    trainingHistory = FitModel(model, 
                               training, 
                               args.learningRate, 
                               args.epochs, 
                               args.batchSize, 
                               validation = validation)
    
    # save history
    if args.trainingHistoryFile is not None:
        SaveHistory(args.trainingHistoryFile, trainingHistory)
        
    # save model
    if args.modelFile is not None:
        model.save(args.modelFile)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
#------------------------------------------------------------------------------