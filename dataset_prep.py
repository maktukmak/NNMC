#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:20:51 2017

@author: mehmetaktukmak
"""

import numpy as np
from numpy import genfromtxt
import pandas as pd

def read_data_set ( data_file_name, data_sheet_name ):

    
    #Reading dataset
    xlsx = pd.ExcelFile(data_file_name)
    X_train = xlsx.parse(data_sheet_name, header=None)
    X_train = X_train.as_matrix()
    
    #Shuffle before training validation split
    np.random.shuffle(X_train)      

    #Detecting nan vlues
    Ymisa = np.isnan(X_train)
    Ymisa = Ymisa.astype(int)

    return X_train, Ymisa;

def normalize_data_set (X_in):
    
    # Normalize the data: subtract the mean image
    mean_train = np.nanmean(X_in, axis = 0, dtype=np.float64)
    std_train = np.nanstd(X_in, axis = 0, dtype=np.float64, ddof = 1)
    X_out = X_in.astype(float)
    X_out -= mean_train
    X_out /= std_train

    #Replacing nan values with 0
    X_out = np.nan_to_num(X_out)
    
    return X_out, mean_train, std_train;

def unnormalize_data_set (X_in, mean, std):
    
    X_out = X_in.astype(float)
    X_out *= std
    X_out += mean

    #Replacing nan values with 0
    X_out = np.nan_to_num(X_out)
    
    return X_out;

def random_data_remove (X_train, nmis, Ymisa):
    #Inducing random missing values to the training set
    X_out = X_train.copy()
    X_miss = np.zeros(X_out.shape)
    n_tot = X_out.shape[0]*X_out.shape[1]
    Y = np.ones(Ymisa.shape)
    Y = Y - Ymisa
    k1 = 0
    cnt_mis = nmis*n_tot - np.sum(Ymisa)
    #cnt_mis = cnt_mis/2

    while (k1 < cnt_mis):
        k2_i = np.random.randint(0,X_out.shape[0])
        k2_j = np.random.randint(0,X_out.shape[1])
        if Y[k2_i, k2_j] == 1:
            k1 += 1
            Y[k2_i, k2_j] = 0
            X_miss[k2_i, k2_j] = X_out[k2_i, k2_j]
            X_out[k2_i, k2_j] = 0
            
#            Y[k2_j, k2_i] = 0
#            X_miss[k2_j, k2_i] = X_out[k2_j, k2_i]
#            X_out[k2_j, k2_i] = 0
#    
    
            
    return X_out, Y, X_miss 