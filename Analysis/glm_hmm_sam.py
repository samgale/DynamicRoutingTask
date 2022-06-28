# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:55:19 2022

@author: svc_ccg
"""

import os
import sys
import numpy as np
import time
from sklearn.model_selection import KFold
from glmhmm import glm_hmm
from glmhmm.utils import permute_states, find_best_fit, compare_top_weights
from glmhmm.visualize import plot_model_params, plot_loglikelihoods, plot_weights
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import fileIO
from DynamicRoutingAnalysisUtils import DynRoutData,sortExps


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

behavDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"
behavFiles = []
while True:
    files = fileIO.getFiles('choose experiments',rootDir=os.path.join(behavDir,'Data'),fileType='*.hdf5')
    if len(files)>0:
        behavFiles.extend(files)
    else:
        break
    
if len(behavFiles)>0:
    exps = []
    for f in behavFiles:
        obj = DynRoutData()
        obj.loadBehavData(f)
        exps.append(obj)
        
exps = sortExps(exps)


y = np.concatenate([obj.trialResponse for obj in exps]).astype(float)

regressors = ['vis1','vis2','sound1','sound2','autoreward','impulsivity']
x = np.zeros((y.size,len(regressors)))
for i,(stim,autoRew) in enumerate(zip(np.concatenate([obj.trialStim for obj in exps]),
                                    np.concatenate([obj.autoRewarded for obj in exps]))):
    if autoRew:
        x[i,4] = 1
    if stim in regressors:
        x[i,regressors.index(stim)] = 1
x[:,-1] = 1

sessions = np.concatenate(([0],np.cumsum([obj.nTrials for obj in exps])))




# hyperparameters
N = y.size # number of data/time points
K = 2 # number of latent states
C = 2 # number of observation classes
D = x.shape[1] # number of GLM inputs (regressors)

# A = transition probabilities
# w = weights
# pi = initial state probabilities

# y = observations (0/1 x n trials)
# x = inputs (n trials x n regressors)
# z = latent states (n trials)

model = glm_hmm.GLMHMM(N,D,C,K,observations="bernoulli",gaussianPrior=1)

inits = 3 # set the number of initializations
maxiter = 250 # maximum number of iterations of EM to allow for each fit
tol = 1e-3

# store values for each initialization
lls_all = np.zeros((inits,250))
A_all = np.zeros((inits,K,K))
w_all = np.zeros((inits,K,D,C))

# fit the model for each initialization
for i in range(inits):
    t0 = time.time()
    # initialize the weights
    A_init,w_init,pi_init = model.generate_params(weights=['GLM',-0.2,1.2,x,y,1])
    # fit the model                     
    lls_all[i,:],A_all[i,:,:],w_all[i,:,:],pi0 = model.fit(y,x,A_init,w_init,maxiter=maxiter,tol=tol,sess=sessions) 
    minutes = (time.time() - t0)/60
    print('initialization %s complete in %.2f minutes' %(i+1, minutes))


















