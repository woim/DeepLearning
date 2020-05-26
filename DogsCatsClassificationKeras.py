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
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, \
    BatchNormalization, Activation, Lambda, Dropout, Add, ZeroPadding2D, AveragePooling2D
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
def CreateModel(dataSize, learningRate, kernelInitializer):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', kernel_initializer=kernelInitializer, input_shape=dataSize))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernelInitializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer=kernelInitializer))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernelInitializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_initializer=kernelInitializer))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernelInitializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1,  activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=learningRate)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])

    return model
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# ResNet identity block
#------------------------------------------------------------------------------
def identity_block(X, f, filters, stage, block, kernelInitializer):
    """
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer=kernelInitializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer=kernelInitializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer=kernelInitializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# ResNet convolutional block
#------------------------------------------------------------------------------
def convolutional_block(X, f, filters, stage, block, kernelInitializer, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer=kernelInitializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer=kernelInitializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer=kernelInitializer)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer=kernelInitializer)(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
        
    return X
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
def CreateResNet(dataSize, learningRate, kernelInitializer):

    # Define the input as a tensor with shape input_shape
    X_input = Input(dataSize)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=kernelInitializer)(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1, kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b', kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c', kernelInitializer=kernelInitializer)

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s=2, kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b',kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c',kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d',kernelInitializer=kernelInitializer)

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s=2, kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b',kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c',kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d',kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e',kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f',kernelInitializer=kernelInitializer)

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s=2,kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b',kernelInitializer=kernelInitializer)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c',kernelInitializer=kernelInitializer)

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2,2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', kernel_initializer = kernelInitializer)(X)
        
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
 
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
    parser.add_argument('-d', '--dataSize',
                        type = int,
                        nargs = '*',
                        default = [1,1,1],
                        help = 'size of the data [n1, n2, n3].')
    parser.add_argument('-t', '--trainingDataFile',
                        type = argparse.FileType('r'),
                        nargs = '?',
                        help = 'numpy file containing training data.')
    parser.add_argument('-v', '--validationDataFile',
                        type  = argparse.FileType('r'),
                        nargs = '?',
                        help = 'numpy file containing validation data.')
    parser.add_argument('-m', '--modelFile',
                        type = str,
                        nargs = '?',
                        help = 'file to store the model.')
    parser.add_argument('-hist', '--trainingHistoryFile',
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
    if args.trainingDataFile is not None:
        trainingData = np.load(args.trainingDataFile, allow_pickle=True)
        training = {
                'data':  trainingData.item().get('data') ,
                'label': trainingData.item().get('label')}
    
    dataSize = args.dataSize
    print("DataSize: {}".format(dataSize))

    # Get validation data
    validation = None
    if args.validationDataFile is not None:
        data = np.load(args.validationDataFile, allow_pickle=True)
        validation = (data.item().get('data'), data.item().get('label'))

    # Create the model
    model = CreateResNet(dataSize, args.learningRate, args.kernelInitializer)
    model.summary()

    # Load Weigths
    if args.weightLoadFile is not None:
        model.load_weights(args.weightLoadFile)

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
