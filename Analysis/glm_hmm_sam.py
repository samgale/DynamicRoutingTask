# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:55:19 2022

@author: svc_ccg
"""

import os
import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import fileIO
from DynamicRoutingAnalysisUtils import DynRoutData,sortExps
import psytrack
from glmhmm import glm_hmm
from glmhmm.utils import permute_states, find_best_fit, compare_top_weights
from glmhmm.visualize import plot_model_params, plot_loglikelihoods, plot_weights



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


regressors = ['vis1','vis2','sound1','sound2','impulsivity']
y = np.concatenate([obj.trialResponse for obj in exps]).astype(float)
x = np.zeros((y.size,len(regressors)))
for i,stim in enumerate(np.concatenate([obj.trialStim for obj in exps])):
    if stim in regressors:
        x[i,regressors.index(stim)] = 1
x[:,-1] = 1

autoRewarded = np.concatenate([obj.autoRewarded for obj in exps])
x = x[~autoRewarded]
y = y[~autoRewarded]

sessionTrials = [obj.nTrials-obj.autoRewarded.sum() for obj in exps]
sessionStartStop = np.concatenate(([0],np.cumsum(sessionTrials)))

sessionBlockTrials = [[np.sum(obj.trialBlock[~obj.autoRewarded]==i) for i in np.unique(obj.trialBlock)] for obj in exps]
blockTrials = np.concatenate(sessionBlockTrials)



# psytrack
d = {'inputs': {key: val[:,None] for key,val in zip(regressors,x.T)},
     'y': y.copy(),
     'dayLength': blockTrials}

weights = {key: 1 for key in d['inputs']}

nWeights = sum(weights.values())

hyper= {'sigInit': 2**4.,
        'sigma': [2**-4.] * nWeights,
        'sigDay': [2**-4.] * nWeights}

optList = ['sigma','sigDay']

hyp, evd, wMode, hess_info = psytrack.hyperOpt(d, hyper, weights, optList)

cvLikelihood,cvProbMiss = psytrack.crossValidate(d, hyper, weights, optList, F=5, seed=0)
cvProbResp = 1-cvProbMiss


fig = plt.figure(figsize=(8,8))
fig.suptitle('mouse '+exps[0].subjectName,fontsize=10)
ylim = [min(0,1.05*wMode.min()),1.05*wMode.max()]
for i in range(len(exps)):
    ax = fig.add_subplot(len(regressors),1,i+1)
    for blockEnd in np.cumsum(sessionBlockTrials[i])[:-1]:
        ax.plot([blockEnd+0.5]*2,ylim,'--',color='0.5')
    for w,lbl,clr in zip(wMode,sorted(weights.keys()),'crgbm'):
        ax.plot(np.arange(sessionTrials[i])+1,w[sessionStartStop[i]:sessionStartStop[i+1]],color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,max(sessionTrials)+1])
    ax.set_ylim(ylim)
    if i==len(exps)-1:
        ax.set_xlabel('trial',fontsize=12)
    else:
        ax.set_xticklabels([])
    if i==0:
        ax.set_ylabel('weights',fontsize=12)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=10)
    ax.set_title('session '+str(i+1),fontsize=10)
plt.tight_layout()

fig = plt.figure(figsize=(8,8))
fig.suptitle('mouse '+exps[0].subjectName,fontsize=10)
ylim = [min(0,1.05*wMode.min()),1.05*wMode.max()]
for i in range(len(exps)):
    ax = fig.add_subplot(len(regressors),1,i+1)
    for stimInd,stim in enumerate(regressors[:-1]):
        sessionInd = slice(sessionStartStop[i],sessionStartStop[i+1])
        trialInd = d['inputs'][stim].astype(bool).squeeze()
        blockStart = 0
        for blockEnd in np.cumsum(sessionBlockTrials[i]):
            if stimInd==0:
                ax.plot([blockEnd+0.5]*2,ylim,'--',color='0.5')
            blockInd = slice(blockStart,blockEnd)
            ax.plot(np.where(trialInd[sessionInd][blockInd])[0]+blockStart+1,gaussian_filter(y[sessionInd][trialInd[sessionInd][blockInd],5),clr)
            blockStart = blockEnd
        ax.plot(np.where(trialInd[sessionInd])[0]+1,cvProbResp[trialInd][sessionInd],':',color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,max(sessionTrials)+1])
    ax.set_ylim(ylim)
    if i==len(exps)-1:
        ax.set_xlabel('trial',fontsize=12)
    else:
        ax.set_xticklabels([])
    if i==0:
        ax.set_ylabel('response prob',fontsize=12)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=10)
    ax.set_title('session '+str(i+1),fontsize=10)
plt.tight_layout()



# glm-hmm
# hyperparameters
N = y.size # number of data/time points
K = 3 # number of latent states
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
    lls_all[i,:],A_all[i,:,:],w_all[i,:,:],pi0 = model.fit(y,x,A_init,w_init,maxiter=maxiter,tol=tol,sess=sessionStartStop) 
    minutes = (time.time() - t0)/60
    print('initialization %s complete in %.2f minutes' %(i+1, minutes))


















