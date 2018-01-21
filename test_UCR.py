# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:24:54 2017

@author: IACJ
"""
import os
from os import path
from os.path import expanduser
from numpy import genfromtxt

import numpy as np

from numpy import array, zeros, argmin, inf 
from numpy import *  
from US_DTW import US_DTW

# classify using kNN  
def kNNClassify_ED(newInput, dataSet, labels, k):  
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row  
  
    ## step 1: calculate Euclidean distance  
    diff = tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise  
    squaredDiff = diff ** 2 # squared for the subtract  
    squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row  
    distance = squaredDist ** 0.5  
  
    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = argsort(distance)  
    
    classCount = {} # define a dictionary (can be append element)  
    for i in range(k):  
        ## step 3: choose the min k distance  
        voteLabel = labels[sortedDistIndices[i]]    
        ## step 4: count the times labels occur  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
  
    ## step 5: the max voted class will return  
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
  
    return maxIndex 

def kNNClassify_DTW(newInput, dataSet, labels, k):  
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row  
  
    ## step 1: calculate Euclidean distance  
    

    distance = np.zeros(numSamples)
    for i in range(numSamples):
        distance[i] =  dtw(newInput,dataSet[i])
  
    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = argsort(distance)  
    
    classCount = {} # define a dictionary (can be append element)  
    for i in range(k):  
        ## step 3: choose the min k distance  
        voteLabel = labels[sortedDistIndices[i]]    
        ## step 4: count the times labels occur  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
  
    ## step 5: the max voted class will return  
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
  
    return maxIndex 

def kNNClassify_US_DTW(newInput, dataSet, labels, k):  
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row  
  
    ## step 1: calculate Euclidean distance  
    

    distance = np.zeros(numSamples)
    for i in range(numSamples):
        us_dtw = US_DTW(newInput,dataSet[i])
        print(i,len(us_dtw.paths))
        distance[i] =  us_dtw.resultDistance
  
    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = argsort(distance)  
    
    classCount = {} # define a dictionary (can be append element)  
    for i in range(k):  
        ## step 3: choose the min k distance  
        voteLabel = labels[sortedDistIndices[i]]    
        ## step 4: count the times labels occur  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
  
    ## step 5: the max voted class will return  
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
  
    return maxIndex 

def dtw(x, y ):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = abs(x[i]-y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    return D1[-1, -1]


# 加载 UCR 数据集的函数
def load_dataset(dataset_name, dataset_folder):
    dataset_path = path.join(dataset_folder, dataset_name)
    train_file_path = path.join(dataset_path, '{}_TRAIN'.format(dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST'.format(dataset_name))

    # training data
    train_raw_arr = genfromtxt(train_file_path, delimiter=',')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = genfromtxt(test_file_path, delimiter=',')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0] - 1

    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':    
    print("Program Begin")
    

    
    ########## 使用 UCR 数据集 ###############
    ucr_dataset_base_folder = expanduser('~/UCR_TS_Archive_2015')
     
    dirs = os.listdir(ucr_dataset_base_folder)
    for dir in dirs:
        print (dir,end=" : \t")
        ucr_dataset_name = dir   
        train_data, train_labels, test_data, test_labels = load_dataset(ucr_dataset_name,ucr_dataset_base_folder)
        print(train_data.shape,train_labels.shape,test_data.shape,test_labels.shape)

    
        ########## 使用 1NN_ED ###################
        Trues = 0
        Falses = 0
        for i in range (test_data.shape[0]):
            x = test_data[i]
            y = test_labels[i]
        
            outputLabel = kNNClassify_ED(x, train_data, train_labels, 1)  
    #        print (i,":\tpredict : ", outputLabel,"\tGroundTruth : ",y,"\t",outputLabel==y)
        
            if (outputLabel==y):
                Trues += 1
            else :
                Falses += 1
        print ("1NN_ED :",Trues/(Trues+Falses))
        ######################################### 
        train_data = np.tile(train_data,2)
        ########## 使用 1NN_US-DTW ###################
        Trues = 0
        Falses = 0
        for i in range (test_data.shape[0]):
            x = test_data[i]
            y = test_labels[i]
            
            
            
            
            
            outputLabel = kNNClassify_US_DTW(x, train_data, train_labels, 1)  
            print (i,":\tpredict : ", outputLabel,"\tGroundTruth : ",y,"\t",outputLabel==y)
        
            if (outputLabel==y):
                Trues += 1
            else :
                Falses += 1
        print ("1NN_DTW :",Trues/(Trues+Falses))
        ################
        
        ########## 使用 1NN_DTW ###################
        Trues = 0
        Falses = 0
        for i in range (test_data.shape[0]):
            x = test_data[i]
            y = test_labels[i]
        
            outputLabel = kNNClassify_DTW(x, train_data, train_labels, 1)  
          #  print (i,":\tpredict : ", outputLabel,"\tGroundTruth : ",y,"\t",outputLabel==y)
        
            if (outputLabel==y):
                Trues += 1
            else :
                Falses += 1
        print ("1NN_DTW :",Trues/(Trues+Falses))
        ######################################### 
        print()
    
    
    