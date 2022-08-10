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


exps = handoffSessions


regressors = ['modality','stimulus','attention','reinforcement','bias']
regressorColors = 'rgcmk'
x = {r: [] for r in regressors}
y = []
sessionTrials = []
sessionBlockTrials = []
sessionStim = []
for obj in exps:
    trials = ~obj.autoRewarded & ~obj.catchTrials
    trialInd = np.where(trials)[0]
    firstTrial = trialInd[0]
    x['modality'].append(np.array([stim[:-1]==rew[:-1] for stim,rew in zip(obj.trialStim[trials],obj.rewardedStim[trials])]))
    x['stimulus'].append(np.array(['1' in stim for stim in obj.trialStim[trials]]))
    x['attention'].append(np.zeros(trials.sum(),dtype=bool))
    x['reinforcement'].append(np.zeros(trials.sum(),dtype=bool))
    for i,stim in enumerate(obj.trialStim[trials]):
        rewardInd = np.where(obj.trialRewarded[:trialInd[i]])[0]
        if len(rewardInd)>0:
            x['attention'][-1][i] = stim[:-1] in obj.trialStim[rewardInd[-1]]
        stimInd = np.where(obj.trialStim[:trialInd[i]]==stim)[0]
        if len(stimInd)>0:
            x['reinforcement'][-1][i] = obj.trialRewarded[stimInd[-1]]
    # x['response'].append(np.concatenate(([obj.trialResponse[firstTrial-1]],obj.trialResponse[trials][:-1])))
    # x['reward'].append(np.concatenate(([obj.trialRewarded[firstTrial-1]],obj.trialRewarded[trials][:-1])))
    x['bias'].append(np.ones(trials.sum(),dtype=bool))
    y.append(obj.trialResponse[trials])
    sessionTrials.append(trials.sum())
    sessionBlockTrials.append(np.array([np.sum(obj.trialBlock[trials]==i) for i in np.unique(obj.trialBlock)]))
    sessionStim.append(obj.trialStim[trials])
    
sessionStartStop = np.concatenate(([0],np.cumsum(sessionTrials)))
blockTrials = np.concatenate(sessionBlockTrials)



# psytrack
holdOutReg = ['all']+regressors
hyperparams = {reg: [] for reg in holdOutReg}
evidence = {reg: [] for reg in holdOutReg}
modelWeights = {reg: [] for reg in holdOutReg}
hessian = {reg: [] for reg in holdOutReg}
cvLikelihood = {reg: [] for reg in holdOutReg}
cvProbNoLick = {reg: [] for reg in holdOutReg}
accuracy = {reg: [] for reg in holdOutReg}
cvFolds = None
for reg in ('all',):#holdOutReg:
    for i in range(len(exps)):
        print(reg,i)
        d = {'inputs': {key: val[i][:,None].astype(float) for key,val in x.items() if key!=reg},
             'y': y[i].astype(float),
             'dayLength': sessionBlockTrials[i]}
        
        weights = {key: 1 for key in d['inputs']}
        
        nWeights = sum(weights.values())
        
        hyper= {'sigInit': 2**4.,
                'sigma': [2**-4.] * nWeights,
                'sigDay': [2**-4.] * nWeights}
        
        optList = ['sigma','sigDay']
        
        hyp, evd, wMode, hess_info = psytrack.hyperOpt(d, hyper, weights, optList)
        hyperparams[reg].append(hyp)
        evidence[reg].append(evd)
        modelWeights[reg].append(wMode)
        hessian[reg].append(hess_info)
        
        if cvFolds is not None:
            cvTrials = d['y'].size - (d['y'].size % cvFolds)
            likelihood,probNoLick = psytrack.crossValidate(psytrack.trim(d,END=cvTrials), hyper, weights, optList, F=cvFolds, seed=0)
            cvLikelihood[reg].append(likelihood)
            cvProbNoLick[reg].append(probNoLick)
            d['y'] -= 1
            accuracy[reg].append(np.abs(d['y'][:cvTrials] - probNoLick))



for i in range(len(exps)):
    w = modelWeights['all'][i]
    wNames = sorted(regressors)
    wColors = [regressorColors[regressors.index(wn)] for wn in wNames]
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ylim = [min(0,1.05*w.min()),1.05*w.max()]
    for blockEnd in np.cumsum(sessionBlockTrials[i])[:-1]:
        ax.plot([blockEnd+0.5]*2,ylim,'k')
    for w,lbl,clr in zip(w,wNames,wColors):
        ax.plot(np.arange(sessionTrials[i])+1,w,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,sessionTrials[i]+1])
    ax.set_ylim(ylim)
    ax.set_xlabel('trial',fontsize=12)
    ax.set_ylabel('weights',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1),fontsize=8)
    ax.set_title(exps[i].subjectName+'_'+exps[i].startTime,fontsize=10)
    plt.tight_layout()


smoothSigma = 5
for i in range(len(exps)):
    probLick = 1-cvProbNoLick['all'][i]
    fig = plt.figure(figsize=(8,8))
    ylim = [-0.05,1.05]
    ax = fig.add_subplot(1,1,1)
    for j,(stim,clr) in enumerate(zip(('vis1','vis2','sound1','sound2'),'rmbc')):
        stimInd = sessionStim[i][:probLick.size] == stim
        blockStart = 0
        smoothedProbResp = []
        for blockEnd in np.cumsum(sessionBlockTrials[i]):
            if j==0:
                ax.plot([blockEnd+0.5]*2,ylim,'k')
            blockInd = slice(blockStart,blockEnd)
            trialInd = stimInd[blockInd]
            smoothedProbResp.append(gaussian_filter(y[i].astype(float)[:probLick.size][blockInd][trialInd],smoothSigma))
            blockStart = blockEnd
        trials = np.where(stimInd)[0]+1
        ax.plot(trials,np.concatenate(smoothedProbResp),color=clr,label=stim+' (mouse)')
        ax.plot(trials,probLick[stimInd],'o',ms=2,mec=clr,mfc='none',label=stim+' (model)')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,sessionTrials[i]+1])
    ax.set_ylim(ylim)
    ax.set_xlabel('trial',fontsize=12)
    ax.set_ylabel('resp prob',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1.5),fontsize=8)
    ax.set_title(exps[i].subjectName+'_'+exps[i].startTime,fontsize=10)
    plt.tight_layout()


for m,lbl in zip((evidence,cvLikelihood,accuracy),('evidence','likelihood','accuracy')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for reg,clr in zip(regressors,regressorColors):
        if lbl=='accuracy':
            a = np.array([np.mean(b) for b in m['all']])
            h = np.array([np.mean(d) for d in m[reg]])
            d = h-a
        else:
            a = np.array(m['all'])
            h = np.array(m[reg])
            d = (h-a)/np.abs(a)
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=reg)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('change in model '+lbl,fontsize=12)
    ax.set_ylabel('cumulative prob',fontsize=12)
    ax.legend()
    plt.tight_layout()
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for reg,clr in zip(regressors,regressorColors):
    wNames = sorted(regressors)
    d = np.array([np.mean(w[wNames.index(reg)]) for w in modelWeights['all']])
    dsort = np.sort(d)
    cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
    ax.plot(dsort,cumProb,color=clr,label=reg)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_ylim([0,1.02])
ax.set_xlabel('model weighting',fontsize=12)
ax.set_ylabel('cumulative prob',fontsize=12)
ax.legend()
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


















