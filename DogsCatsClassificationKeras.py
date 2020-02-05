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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, \
    BatchNormalization, Activation, Lambda, Dropout
from keras.models import Model


#------------------------------------------------------------------------------
# Save function
#------------------------------------------------------------------------------
def SaveHistory(csvFilename, history):
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = csvFilename
    with open(hist_csv_file, mode='w', newline='\n') as f:
        hist_df.to_csv(f, float_format='%.12f')
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Callback
#------------------------------------------------------------------------------
class ActivationHistory(keras.callbacks.Callback):

    def __init__(self):
        self.layersIndex = []
        self.history     = {}

    def set_training(self, training):
        self.trainingData = np.array(training)

    def set_layers(self, layersIndex):
        self.layersIndex = layersIndex
        for index in self.layersIndex:
            self.history[self.get_layer_name(index)] = {'mean': [], 'std': []}

    def get_layer_name(self, index):
        return 'layer' + str(index)

    def format_history(self):
        formatted_history = {}
        for index in self.layersIndex:
            formatted_history[self.get_layer_name(index) + '_mean'] = \
                self.history[self.get_layer_name(index)]['mean']
            formatted_history[self.get_layer_name(index) + '_std']  = \
                self.history[self.get_layer_name(index)]['std']
        return formatted_history

    def on_epoch_end(self, epoch, logs=None):
        for index in self.layersIndex:
            intermediate_layer_model = Model(inputs  = self.model.input,
                                             outputs = self.model.layers[index].output)
            intermediate_output = intermediate_layer_model.predict(self.trainingData)
            mean = np.mean(intermediate_output)
            std  = np.std(intermediate_output)
            self.history[self.get_layer_name(index)]['mean'].append(mean)
            self.history[self.get_layer_name(index)]['std'].append(std)
            #print('Epoch {} layer {}: mean = {} std = {}'.format(epoch, self.get_layer_name(index), mean, std))
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# CNN creation
#------------------------------------------------------------------------------
def CreateModel(dataSize, learningRate, kernelInitializer):#, batchNormalization):

    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, activation='relu',
                     kernel_initializer=kernelInitializer, input_shape=dataSize))
    model.add(BatchNormalization())
    # model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu',
    #                  kernel_initializer=kernelInitializer))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=3, activation='relu', kernel_initializer=kernelInitializer))
    model.add(BatchNormalization())
    # model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernelInitializer))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer=kernelInitializer))
    model.add(BatchNormalization())
    # model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernelInitializer))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_initializer=kernelInitializer))
    # model.add(BatchNormalization())
    # model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernelInitializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # model.add(Dense(206, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(1,  activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=learningRate)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])

    return model
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Fit the model
#------------------------------------------------------------------------------
def FitModel(
        model,
        training,
        epochs,
        startingEpoch,
        batchSize,
        shuffle,
        layersIndex,
        verbose = 0,
        validation = None):

    monitor = []
    if layersIndex is not None:
        activationHistory = ActivationHistory()
        activationHistory.set_training(training['data'])
        activationHistory.set_layers(layersIndex)
        monitor.append(activationHistory)

    history = model.fit(training['data'],
                        training['label'],
                        epochs = epochs,
                        verbose = verbose,
                        batch_size = batchSize,
                        shuffle = shuffle,
                        validation_data = validation,
                        callbacks = monitor)

    if layersIndex is not None:
        activationHistory = monitor[0].format_history()
        for key in activationHistory:
            history.history[key] = activationHistory[key]

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
                        default = 1e-4,
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
    parser.add_argument('-ki','--kernelInitializer',
                        type = str,
                        default = 'he_normal',
                        help = 'Choose kernel initializer proposed by keras.')
    parser.add_argument('-li', '--layersIndex',
                        type = int,
                        nargs = '*',
                        help = 'list of layers index to monitor during training.')
    parser.add_argument('--dryRun',
                        action = 'store_true',
                        help = 'dry run without any training.')
#    parser.add_argument('-bNorm', '--batchNormalization',
#                        type = str,
#                        nargs = '?',
#                        default = None,
#                        choices = ['preActivation', 'postActivation'],
#                        help = 'Perform batch normalization before activation function.')
    args = parser.parse_args()

    # Get training data shape
    trainingData = np.load(args.trainingDataFile, allow_pickle=True)
    training = {
            'data':  trainingData.item().get('data') ,
            'label': trainingData.item().get('label')}
    dataSize = training['data'][0].shape

    # Get validation data
    validation = None
    if args.validationDataFile is not None:
        data = np.load(args.validationDataFile, allow_pickle=True)
        validation = (data.item().get('data'), data.item().get('label'))

    # Create the model
    model = CreateModel(dataSize,
                        args.learningRate,
                        args.kernelInitializer)

    # Load Weigths
    if args.weightLoadFile is not None:
        model.load_weights(args.weightLoadFile)

    # Print the model
    model.summary()

    # Fit model
    shuffle = not args.sequential
    #args.batchNormalization
    if not args.dryRun:
        trainingHistory = FitModel(model,
                                   training,
                                   args.epochs,
                                   args.startingEpoch,
                                   args.batchSize,
                                   shuffle,
                                   args.layersIndex,
                                   0,
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
