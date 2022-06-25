# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:55:19 2022

@author: svc_ccg
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import KFold
from glmhmm import glm_hmm
from glmhmm.utils import permute_states, find_best_fit, compare_top_weights
from glmhmm.visualize import plot_model_params, plot_loglikelihoods, plot_weights

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
font = {'family'   : 'sans-serif',
        'sans-serif' : 'Helvetica',
        'weight'   : 'regular',
        'size'     : 18}
mpl.rc('font', **font)


# hyperparameters
N = 50000 # number of data/time points
K = 2 # number of latent states
C = 3 # number of observation classes
D = 4 # number of GLM inputs (regressors)

# A = transition probabilities
# w = weights
# pi = initial state probabilities
true_GLMHMM = glm_hmm.GLMHMM(N,D,C,K,observations="multinomial")
A_true,w_true,pi_true = true_GLMHMM.generate_params(weights=['uniform',-1,1,1])

# y = observations
# x = inputs
# z = latent states
# true_y,true_z,true_x = true_GLMHMM.generate_data(A_true,w_true)

# Ethan's mice
# 594825 – 4/11-4/15
# 596921 – 3/29-4/1
# 589583 – 4/05-4/11
# 588997 – 3/9-3/15

# Sam's mice
# 594530:  2/25,28; 3/1-4
# 596919:  2/25,28; 3/1
# 596926:  2/25,28: 3/1,9,11
# 610739:  3/2-4,7-9




# fit the model
inits = 2 # set the number of initializations

# store values for each initialization
lls_all = np.zeros((inits,250))
A_all = np.zeros((inits,K,K))
w_all = np.zeros((inits,K,D,C))

# fit the model for each initialization
for i in range(inits):
    A_init,w_init,pi_init = true_GLMHMM.generate_params() # initialize the model parameters
    lls_all[i,:],A_all[i,:,:],w_all[i,:,:],pi0 = true_GLMHMM.fit(true_y,true_x,A_init,w_init) # fit the model
    print('initialization %s complete' %(i+1))


















