# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:55:02 2019

@author: francois
"""
import argparse
import os
import numpy as np
import pandas as pd

from skimage import io
from skimage import transform

from sklearn.utils import shuffle

import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
# Read image
#------------------------------------------------------------------------------
def ReadImage(filename, size):
    image = io.imread(filename, as_gray=True)
    data = transform.resize(image, (size,size), anti_aliasing=True)
    data /= 255
    return data
#------------------------------------------------------------------------------
    

#------------------------------------------------------------------------------
# Loading data
#------------------------------------------------------------------------------
def LoadData(dataDir, maxData, size, percentageSplit):
       
    files = shuffle(os.listdir(dataDir), random_state=0)
    countCats = 0
    countDogs = 0
    trainingcount   = 0
    validationCount = 0
    count = 0
    
    trainingSize    = int(maxData*(1-percentageSplit))
    validationSize  = int(maxData*percentageSplit)
    trainData       = np.zeros((trainingSize, size, size))
    trainLabel      = np.zeros(trainingSize)
    validationData  = np.zeros((validationSize, size, size))
    validationLabel = np.zeros(validationSize)

    print("numberCats: {}".format(maxData/2))
    print("numberDogs: {}".format(maxData/2))
    print("trainingSize: {}".format(trainingSize))
    print("validationSize: {}".format(validationSize))
    
    while countCats + countDogs < maxData:
        filename = files[count]
        
        if filename.startswith("cat") and countCats < maxData/2 :
            data = ReadImage(os.path.join(dataDir,filename), size) 
            if countCats < trainingSize/2:
                trainLabel[trainingcount] = 1
                trainData[trainingcount] = data
                trainingcount += 1
            else:
                validationLabel[validationCount] = 1
                validationData[validationCount] = data
                validationCount += 1
            countCats += 1
            
        elif filename.startswith("dog") and countDogs < maxData/2 :
            data = ReadImage(os.path.join(dataDir,filename), size) 
            if countDogs < trainingSize/2:
                trainData[trainingcount] = data
                trainingcount += 1
            else:
                validationData[validationCount] = data
                validationCount += 1
            countDogs += 1

        count += 1
        print('\rcats: {}% \tDogs: {}%'.format(100*countCats/(maxData/2), 100*countDogs/(maxData/2)), end="")
    
    print()
    trainData = trainData.reshape(-1, size, size, 1)
    validationData = validationData.reshape(-1, size, size, 1)
    
    return {'data': trainData, 'label':trainLabel}, {'data': validationData, 'label':validationLabel }
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Loading data in dataframe
#------------------------------------------------------------------------------
def LoadDataInDF(dataDir, maxData, percentageSplit):
       
    files           = shuffle(os.listdir(dataDir), random_state=0)
    countCats       = 0
    countDogs       = 0
    trainingcount   = 0
    validationCount = 0
    count           = 0
    
    trainingSize    = int(maxData*(1-percentageSplit))
    validationSize  = int(maxData - trainingSize)
    trainData       = []
    trainLabel      = []
    validationData  = []
    validationLabel = []

    print("numberCats: {}".format(maxData/2))
    print("numberDogs: {}".format(maxData/2))
    print("trainingSize: {}".format(trainingSize))
    print("validationSize: {}".format(validationSize))
    
    while countCats + countDogs < maxData:
        filename = files[count]        
        if filename.startswith("cat") and countCats < maxData/2 :
            if countCats < trainingSize/2:
                trainLabel.append(1)
                trainData.append(filename)
                trainingcount += 1
            else:
                validationLabel.append(1)
                validationData.append(filename)
                validationCount += 1
            countCats += 1
            
        elif filename.startswith("dog") and countDogs < maxData/2 :
            if countDogs < trainingSize/2:
                trainLabel.append(0)
                trainData.append(filename)
                trainingcount += 1
            else:
                validationLabel.append(0)
                validationData.append(filename)
                validationCount += 1
            countDogs += 1

        count += 1
        print('\rcats: {}% \tDogs: {}%'.format(100*countCats/(maxData/2), 100*countDogs/(maxData/2)), end="")
    
    print()
 
    trainingData = pd.DataFrame({'filename': trainData, 'class': trainLabel})
    testingData = pd.DataFrame({'filename': validationData, 'class': validationLabel})

    return trainingData, testingData
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Processing data
#------------------------------------------------------------------------------
def ProcessData(training, validation):
    
    meanTraining = np.mean(training['data'], axis=0)    
    training['mean'] = meanTraining
    centeredTraining = training['data'] - training['mean']
    centeredValidation = validation['data'] - training['mean']
    
    training['data'] = centeredTraining
    validation['data'] = centeredValidation
    
    return training, validation    
#------------------------------------------------------------------------------
    

#------------------------------------------------------------------------------
# Convert data to image
#------------------------------------------------------------------------------
def ConvertToImage(data):
    image = np.reshape(data,(data.shape[0],data.shape[1]))
    return image
#------------------------------------------------------------------------------
    

#------------------------------------------------------------------------------
# Processing data
#------------------------------------------------------------------------------
def PlotSample(training, validation, title, trainingIndex = None, validationIndex = None):
    
    NumberSamples = 3
    if trainingIndex is None:
        trainingIndex = np.random.randint(0, training['data'].shape[0], size = NumberSamples)
    if validationIndex is None:
        validationIndex = np.random.randint(0, validation['data'].shape[0], size = NumberSamples)
    
    fig, axarr = plt.subplots(2, NumberSamples)   
    fig.suptitle(title)
    for i in np.arange(NumberSamples):
        trainImage = ConvertToImage(training['data'][trainingIndex[i]])
        validationImage = ConvertToImage(validation['data'][validationIndex[i]])
        axarr[0, i].imshow(trainImage, cmap=plt.cm.gray)
        axarr[0, i].set_title('training image ' + str(trainingIndex[i]) )
        axarr[1, i].imshow(validationImage, cmap=plt.cm.gray)
        axarr[1, i].set_title('validation image ' + str(validationIndex[i]) )
    
    plt.show()
    
    return trainingIndex, validationIndex
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Application
#------------------------------------------------------------------------------
def main():    
    parser = argparse.ArgumentParser(
        description = 'Preprocess Data script',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataDir',
                        type = str,
                        help = 'directory containing data.')
    parser.add_argument('trainingDataFile',
                        type = str,
                        help = 'numpy file containing training data.')
    parser.add_argument('-v', '--validationDataFile',
                        type = str,
                        nargs = '?',
                        help = 'numpy file containing validation data.')    
    parser.add_argument('-m', '--maxData',
                        type = int,
                        help = 'Number of run on the data.')
    parser.add_argument('-s', '--size',
                        type = int,
                        default = 64,
                        help = 'Number of run on the data.')
    parser.add_argument('-p', '--precentageSplit',
                        type = float,
                        default = 0.2,
                        help = 'Percentage split between training and validation.')
    parser.add_argument('-d', '--dataFrame',
                        action = 'store_true',
                        help = 'store the data as data frame.')
    args = parser.parse_args()
	    
    if args.maxData == None:
        args.maxData = len(os.listdir(args.dataDir))

    if args.dataFrame == True:

        training, validation = LoadDataInDF(args.dataDir, args.maxData, args.precentageSplit)
        training.to_csv(args.trainingDataFile, index=False)
        if args.validationDataFile != None:
            validation.to_csv(args.validationDataFile, index=False)

    else:
        
        training, validation = LoadData(args.dataDir, args.maxData, args.size, args.precentageSplit)
        trainingIndex, validationIndex = PlotSample(training, validation, "Original data")
        training, validation = ProcessData(training, validation)
        PlotSample(training, validation, "Processed data", trainingIndex, validationIndex)
    
        np.save(args.trainingDataFile, training)
        if args.validationDataFile != None:
            np.save(args.validationDataFile, validation)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
#------------------------------------------------------------------------------