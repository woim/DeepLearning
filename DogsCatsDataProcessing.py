# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:55:02 2019

@author: francois
"""
import argparse
import os
import numpy as np

from skimage import io
from skimage import transform

from sklearn.utils import shuffle

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
# Processing data
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
# Application
#------------------------------------------------------------------------------
def main():    
    parser = argparse.ArgumentParser(
        description = 'Preprcoss Data script',
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
    args = parser.parse_args()
	    
    if args.maxData == None:
        args.maxData = len(os.listdir(args.dataDir))
    
    training, validation = LoadData(args.dataDir, args.maxData, args.size, args.precentageSplit)
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