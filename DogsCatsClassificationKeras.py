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
    with open(hist_csv_file, mode='w', newline='\n') as f:
        hist_df.to_csv(f, float_format='%.12f')
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# CNN creation
#------------------------------------------------------------------------------
def CreateModel(dataSize, learningRate):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=dataSize))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=learningRate)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return model
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Fit the model
#------------------------------------------------------------------------------
def FitModel(model, training, epochs, startingEpoch, batchSize, shuffle, validation = None):
    
    history = model.fit(training['data'],
                        training['label'], 
                        epochs = epochs, 
                        verbose = 2,
                        batch_size = batchSize,
                        shuffle = shuffle,
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
                        default = 100,
                        help = 'Number of epochs for trainig data.')
    parser.add_argument('-se', '--startingEpoch',
                        type = int,
                        nargs = '?',
                        default = 0,
                        help = 'Starting epoch (useful to resume).')
    parser.add_argument('-lr', '--learningRate',
                        type = float,
                        default = 1e-3,
                        help = 'Learning rate for training.')
    parser.add_argument('-b', '--batchSize',
                        type = int,
                        default = 64,
                        help = 'Learning rate for training.')
    parser.add_argument('-seq', '--sequential',
                        action = 'store_true',
                        help = 'train sequentially on the batch.')
    parser.add_argument('-lw', '--weightLoadFile',
                        type = str,
                        nargs = '?',
                        help = 'file from which to load the Weights.')
    args = parser.parse_args()
    
    # Get training data shape
    trainingData = np.load(args.trainingDataFile)
    training = {
            'data':  trainingData.item().get('data') ,
            'label': trainingData.item().get('label')}
    dataSize = training['data'][0].shape
    
    # Get validation data
    validation = None
    if args.validationDataFile is not None:
        data = np.load(args.validationDataFile)
        validation = (data.item().get('data'), data.item().get('label'))
        
    # Create the model
    model = CreateModel(dataSize, args.learningRate)
    
    # Load Weigths
    if args.weightLoadFile is not None:
        model.load_weights(args.weightLoadFile)
        
    # Print the model    
    model.summary()
    
    # Fit model
    shuffle = not args.sequential
    trainingHistory = FitModel(model, 
                               training,                                
                               args.epochs, 
                               args.startingEpoch,
                               args.batchSize,
                               shuffle,
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