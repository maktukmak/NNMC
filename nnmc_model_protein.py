#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:02:17 2017

@author: mehmetaktukmak
"""
import numpy as np
from dataset_prep import random_data_remove
from dataset_prep import read_data_set
from dataset_prep import normalize_data_set
from nnmc_model import multimodal_alg

class exp_params:   # User changable parameters
    def __init__(self):
        self.fraction_of_missings = 0.1	# Fraction of values to be removed

class opt_params:	# User changable parameters
    def __init__(self):
        self.reg = 0.1	# Regularization strength
        self.no_of_hidden_layers = 5	# Number of neurons in hidden layer

class fixed_params:	# Fixed system parameters 
    def __init__(self):
        self.iteration_outer_loop = 50
        self.min_iteration_outer_loop = 40	
        self.outer_learning_rate = 0.1	
        self.plot_outer_loop_on_off = 0	
        self.tf_second_pass = 50
        self.max_iter_nn_train = 500
        self.min_iter_nn_train = 400
        self.step_size = 0.1
        self.plt_inner_loop_on_off = 0
        self.plot_inner_loop_feature_no = 0
        self.neural_net_store_on_off = 1
        self.shuffle_before_iterate = 1

exp_params = exp_params()
opt_params = opt_params()
fixed_params = fixed_params()

# Load dataset
X_train, Ymisa, = read_data_set ('protein.xlsx', 'protein_sh')

# Normalize
X_train_norm,_,_ = normalize_data_set(X_train)
    
print( "INFO: Dataset loaded successfully!")
print( "INFO: Sample size = {:d}, Feature size = {:d}" .format(X_train.shape[0], X_train.shape[1]))
print( "INFO: {:.1f} percent of values are missing in the dataset inherently" .format(np.count_nonzero(Ymisa)/(X_train.shape[0]*X_train.shape[1])))

# Remove values randomly
X_train_modify, Y, X_miss = random_data_remove (X_train_norm, exp_params.fraction_of_missings, Ymisa)
print( "INFO: {:f} percentage of data removed" .format(exp_params.fraction_of_missings))
print( "INFO: Feature based algorithm will be applied" )

# Run NNMC
Mn = np.ones(X_train_modify.shape[1], dtype=int)
XNN_out = multimodal_alg (X_train_modify, Y, X_miss, Mn, opt_params, fixed_params)

# Evaluate MSE
miss_glob = np.where(Y == 0)        
MSE = np.sum(np.power((XNN_out[miss_glob] - X_miss[miss_glob]), 2)) / miss_glob[0].size
print("MSE = %f" %MSE)


