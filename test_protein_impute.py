#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:02:17 2017

@author: mehmetaktukmak
"""
import numpy as np
from utils import random_data_remove
from utils import read_data_set
from utils import normalize
from nnmc_model import nnmc_model

fraction_of_missings = 0.1	# Fraction of values to be removed

# Load dataset
X, Miss, = read_data_set('protein.xlsx', 'DS5')

# Normalize
X_norm,_,_ = normalize(X)
    
print( "INFO: Dataset loaded successfully!")
print( "INFO: Sample size = {:d}, Feature size = {:d}" .format(X.shape[0], X.shape[1]))
print( "INFO: {:.1f}% of values are missing in the dataset" .format(np.count_nonzero(Miss)*100/(X.shape[0]*X.shape[1])))

# Remove values randomly
X_train, X_mask, X_miss = random_data_remove(X_norm, fraction_of_missings, Miss)
print( "INFO: {:.1f}% of data removed" .format(fraction_of_missings*100))

# Impute missing values with NNMC model
model = nnmc_model()
X_out = model.mc_complete(X_train, X_mask, X_miss)

# Evaluate MSE
miss_glob = np.where(X_mask == 0)        
MSE = np.sum(np.power((X_out[miss_glob] - X_miss[miss_glob]), 2)) / miss_glob[0].size
print("MSE = %f" %MSE)