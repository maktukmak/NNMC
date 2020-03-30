#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:20:51 2017

@author: mehmetaktukmak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

def read_data_set ( data_file_name, data_sheet_name ):

    #Reading dataset
    xlsx = pd.ExcelFile(data_file_name)
    X = xlsx.parse(data_sheet_name, header=None)
    X = X.as_matrix()
    
    #Shuffle before training validation split
    np.random.shuffle(X)      

    #Detecting nan vlues
    Miss = np.isnan(X).astype(int)
    
    return X, Miss;

def normalize(X):
    
    # Normalize the data: subtract the mean image
    mean = np.nanmean(X, axis = 0, dtype=np.float64)
    std = np.nanstd(X, axis = 0, dtype=np.float64, ddof = 1)
    X_norm = X.astype(float)
    X_norm -= mean
    X_norm /= std

    #Replacing nan values with 0
    X_norm = np.nan_to_num(X_norm)
    
    return X_norm, mean, std;

def unnormalize_data_set (X_in, mean, std):
    
    X_out = X_in.astype(float)
    X_out *= std
    X_out += mean

    #Replacing nan values with 0
    X_out = np.nan_to_num(X_out)
    
    return X_out;

def random_data_remove (X, nmis, Ymisa):
    #Inducing random missing values to the training set
    X_out = X.copy()
    X_miss = np.zeros(X_out.shape)
    n_tot = X_out.shape[0]*X_out.shape[1]
    Y = np.ones(Ymisa.shape)
    Y = Y - Ymisa
    k1 = 0
    cnt_mis = nmis*n_tot - np.sum(Ymisa)

    while (k1 < cnt_mis):
        k2_i = np.random.randint(0,X_out.shape[0])
        k2_j = np.random.randint(0,X_out.shape[1])
        if Y[k2_i, k2_j] == 1:
            k1 += 1
            Y[k2_i, k2_j] = 0
            X_miss[k2_i, k2_j] = X_out[k2_i, k2_j]
            X_out[k2_i, k2_j] = 0
            
    return X_out, Y, X_miss 

def plot_tsne (X, labels):
    
    no_of_samples = 1000
    
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    tsne_y = tsne.fit_transform(X[0:no_of_samples])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_y[:, 0], tsne_y[:, 1], tsne_y[:, 1], c=labels[0:no_of_samples])
    plt.savefig('tsne_vis.pdf', format='pdf')
    plt.show()

def plt_outer_loop (MSE):
    
    plt.plot(MSE, label='MSE')
    #plt.plot(update_norm, label='Cross Entropy')
    plt.xlabel('Number of iterations')
    plt.legend(bbox_to_anchor=(1.00, 1), loc=1, borderaxespad=0.)
    plt.savefig('Outer_loop.pdf', format='pdf')
    plt.show()
    
def plt_inner_loop (val_error_function,  loss_function, i):
    
    plt.plot(val_error_function, label='Validation error')
    plt.plot(loss_function, label='Loss function')
    plt.ylabel('MSE values for row %d missing value predictions'% i)
    plt.xlabel('Number of epochs')
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)
    plt.ylim((0,1))
    plt.show()
