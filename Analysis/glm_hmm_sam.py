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
import ssm



regressors = ['context','reinforcement','reward','bias']
regressorColors = 'rgbk'
x = {r: [] for r in regressors}
y = []
sessionTrials = []
sessionBlockTrials = []
sessionStim = []
sessionRewardStim = []

trainingPhase = 'after learning'
modelType = 'mixedAgentRL'
d = modelData[trainingPhase]
mouse = '699238'

for session in d[mouse]:
    obj = sessionData[trainingPhase][mouse][session]
    for reg in regressors[:-1]:
        betaAction,biasAction,biasAttention,visConfidence,audConfidence,wContext,alphaContext,decayContext,alphaReinforcement,wReward,alphaReward,wPerseveration,alphaPerseveration = d[mouse][session][modelType]['params'][0]
        if reg == 'context':
            wContext = 1
            wReward = 0
        elif reg == 'reinforcement':
            wContext = 0
            alphaContext = 0
            wReward = 0
        elif reg == 'reward':
            wContext = 0
            alphaContext = 0
            wReward = 1
        params = (betaAction,biasAction,biasAttention,visConfidence,audConfidence,wContext,alphaContext,decayContext,alphaReinforcement,wReward,alphaReward,wPerseveration,alphaPerseveration)
        x[reg].append(runModel(obj,*params,**modelTypeParams[modelType])[-2][0])
    x['bias'].append(np.ones(obj.nTrials))
    y.append(obj.trialResponse)
    sessionTrials.append(obj.nTrials)
    sessionBlockTrials.append(np.array([np.sum(obj.trialBlock==i) for i in np.unique(obj.trialBlock)]))
    sessionStim.append(obj.trialStim) 
    sessionRewardStim.append(obj.rewardedStim)
sessionStartStop = np.concatenate(([0],np.cumsum(sessionTrials)))
blockTrials = np.concatenate(sessionBlockTrials)

regressorCorr = np.zeros((len(regressors)-1,)*2)
for i,ri in enumerate(regressors[:-1]):
    for j,rj in enumerate(regressors[:-1]):
        regressorCorr[i,j] = np.corrcoef(np.concatenate(x[ri]),np.concatenate(x[rj]))[0,1]



## psytrack
holdOut = ['none']
holdOutColors = ('0.5',)
hyperparams = {reg: [] for reg in holdOut}
evidence = {reg: [] for reg in holdOut}
modelWeights = {reg: [] for reg in holdOut}
hessian = {reg: [] for reg in holdOut}
cvLikelihood = {reg: [] for reg in holdOut}
cvProbNoLick = {reg: [] for reg in holdOut}
accuracy = {reg: [] for reg in holdOut}
cvFolds = 10
for reg in holdOut:
    d = {'inputs': {key: np.concatenate(val)[:,None] for key,val in x.items() if key!=reg and key not in reg},
         'y': np.concatenate(y).astype(float),
         'dayLength': np.array(sessionTrials)}
    
    weights = {key: 1 for key in d['inputs']}
    
    nWeights = sum(weights.values())
    
    hyper= {'sigInit': 2**4.,
            'sigma': [2**-4.] * nWeights,
            'sigDay': [2**-4.] * nWeights}
    
    optList = ['sigma','sigDay']
    
    try:
        hyp, evd, wMode, hess_info = psytrack.hyperOpt(d, hyper, weights, optList)
        hyperparams[reg].append(hyp)
        evidence[reg].append(evd)
        modelWeights[reg].append(wMode)
        hessian[reg].append(hess_info)
    except:
        print('\nerror fitting ',reg,i)
        hyperparams[reg].append(None)
        evidence[reg].append(np.nan)
        modelWeights[reg].append(np.full((nWeights,d['y'].size),np.nan))
        hessian[reg].append(None)
    
    if cvFolds is not None:
        cvTrials = d['y'].size - (d['y'].size % cvFolds)
        likelihood = np.nan
        probNoLick = np.full(cvTrials,np.nan)
        if hyperparams[reg][-1] is not None:
            try:
                likelihood,probNoLick = psytrack.crossValidate(psytrack.trim(d,END=cvTrials), hyper, weights, optList, F=cvFolds, seed=0)
            except:
                print('\nerror cross validating ',reg, i)
        cvLikelihood[reg].append(likelihood)
        cvProbNoLick[reg].append(probNoLick)
        d['y'] -= 1
        accuracy[reg].append(np.abs(d['y'][:cvTrials] - probNoLick))



preTrials = postTrials = 45
x = np.arange(-preTrials,postTrials+1)
for ho in holdOut:
    title = 'all regressors' if ho=='none' else 'no '+ho+' regressor'
    for blockType,rewardStim in zip(('visual rewarded','auditory rewarded'),('vis1','sound1')):
        visGoRespProb = []
        visNogoRespProb = []
        soundGoRespProb = []
        soundNogoRespProb = []
        weights = []
        for i in range(len(exps)):
            blockStartStop = np.cumsum(sessionBlockTrials[i])
            lickProb = 1-cvProbNoLick[ho][i]
            mWeights = modelWeights[ho][i]
            for j,(blockStart,blockStop) in enumerate(zip(blockStartStop[:-1],blockStartStop[1:])):
                if sessionRewardStim[i][blockStart]==rewardStim:
                    for rp,stim in zip((visGoRespProb,visNogoRespProb,soundGoRespProb,soundNogoRespProb),('vis1','vis2','sound1','sound2')):
                        stimTrials = sessionStim[i][:lickProb.size]==stim
                        r = np.full(preTrials+postTrials+1,np.nan)
                        r[:preTrials] = lickProb[:blockStart][stimTrials[:blockStart]][-preTrials:]
                        post = lickProb[blockStart:][stimTrials[blockStart:]][:postTrials]
                        r[preTrials+1:preTrials+1+post.size] = post
                        rp.append(r)
                    w = np.full((mWeights.shape[0],preTrials+postTrials+1),np.nan)
                    for k,mw in enumerate(mWeights):
                        w[k,:preTrials] = mw[blockStart-preTrials:blockStart]
                        w[k,preTrials+1:] = mw[blockStart:blockStart+postTrials]
                    weights.append(w)
            
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ylim = [0,1.01]
        ax.plot([0,0],ylim,'k--')
        for d,clr,ls,lbl in zip((visGoRespProb,visNogoRespProb,soundGoRespProb,soundNogoRespProb),'ggmm',('-','--','-','--'),('visual go','visual nogo','auditory go','auditory nogo')):
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=lbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim(ylim)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Cross-Validated Response Probability')
        ax.set_title(title+'\n'+blockType)
        # ax.legend(loc='lower right')
        plt.tight_layout()
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        wNames = sorted([r for r in regressors if r!=ho and r not in ho])
        wColors = [regressorColors[regressors.index(wn)] for wn in wNames]
        wMean = np.nanmean(weights,axis=0)
        wSem = np.nanstd(weights,axis=0)/(np.sum(~np.isnan(weights),axis=0)**0.5)
        for m,s,clr,lbl in zip(wMean,wSem,wColors,wNames):
            lbl = 'preservation' if lbl=='persistence' else lbl
            ax.plot(x,m,clr,label=lbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Regressor Weight')
        ax.set_title(title+'\n'+blockType)
        ax.legend(loc='lower right')
        plt.tight_layout()


for m,lbl in zip((evidence,cvLikelihood,accuracy),('evidence','likelihood','accuracy')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for reg,clr in zip(holdOut[1:],holdOutColors[1:]):
        if lbl=='accuracy':
            a = np.array([np.mean(b) for b in m['none']])
            h = np.array([np.mean(d) for d in m[reg]])
        else:
            a = np.array(m['none'])
            h = np.array(m[reg])
        d = h-a
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
for reg,clr in zip(holdOut,holdOutColors):
    d = np.array([np.mean(a) for a in accuracy[reg]])
    dsort = np.sort(d)
    cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
    ax.plot(dsort,cumProb,color=clr,label=reg)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_ylim([0,1.02])
ax.set_xlabel('accuracy',fontsize=12)
ax.set_ylabel('cumulative prob',fontsize=12)
ax.legend()
plt.tight_layout()
    

for ho in holdOut:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    r,c = zip(*[(r,c) for r,c in zip(regressors,regressorColors) if r!=ho and r not in ho])
    wNames = sorted(r)
    for reg,clr in zip(r,c):
        d = np.array([np.nanmean(w[wNames.index(reg)]) for w in modelWeights[ho]])
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=reg)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('regressor weight',fontsize=12)
    ax.set_ylabel('cumulative fraction of sessions',fontsize=12)
    # ax.set_title(ho)
    ax.legend()
    plt.tight_layout()
    

for ho in holdOut:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    r,c = zip(*[(r,c) for r,c in zip(regressors,regressorColors) if r!=ho and r not in ho])
    wNames = sorted(r)
    for reg,clr in zip(r,c):
        d = []
        for i,w in enumerate(modelWeights[ho]):
            blockStartStop = np.concatenate(([0],np.cumsum(sessionBlockTrials[i])))
            for blockStart,blockStop in zip(blockStartStop[:-1],blockStartStop[1:]):
                d.append(np.nanmean(w[wNames.index(reg)][blockStart:blockStop]))
        d = np.array(d)
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=reg)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('regressor weight',fontsize=12)
    ax.set_ylabel('cumulative fraction of blocks',fontsize=12)
    # ax.set_title(ho)
    ax.legend()
    plt.tight_layout()
    

for ho in holdOut:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    r,c = zip(*[(r,c) for r,c in zip(regressors,regressorColors) if r!=ho and r not in ho])
    wNames = sorted(r)
    for reg,clr in zip(r,c):
        d = np.array([np.nanmean(np.concatenate([modelWeights[ho][i][wNames.index(reg)] for i in np.where(mouseIds==mid)[0]])) for mid in np.unique(mouseIds)])
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=reg)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('model weighting',fontsize=12)
    ax.set_ylabel('cumulative fraction of mice',fontsize=12)
    # ax.set_title(ho)
    ax.legend()
    plt.tight_layout()


alim = [-4,14.5]
for ho in ('none',):#holdOut:
    reg = [r for r in regressors if r!=ho and r not in ho]
    wNames = sorted(reg)    
    if 'model' in wNames and 'reinforcement' in wNames:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        mod,reinf = [[np.mean(w[wNames.index(r)]) for w in modelWeights[ho]] for r in ('model','reinforcement')]
        ax.plot(mod,reinf,'ko',label='sessions')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_xlabel('model regressor weight',fontsize=12)
        ax.set_ylabel('reinforcment regressor weight',fontsize=12)
        ax.legend()
        # ax.set_title(ho)
        plt.tight_layout()
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        mod = []
        reinf = []
        modFirst = []
        reinfFirst = []
        for reg,d,df in zip(('model','reinforcement'),(mod,reinf),(modFirst,reinfFirst)):
            for i,w in enumerate(modelWeights[ho]):
                blockStartStop = np.concatenate(([0],np.cumsum(sessionBlockTrials[i])))
                for j,(blockStart,blockStop) in enumerate(zip(blockStartStop[:-1],blockStartStop[1:])):
                    d.append(np.nanmean(w[wNames.index(reg)][blockStart:blockStop]))
                    if j==0:
                        df.append(d[-1])
        ax.plot(mod,reinf,'ko',label='blocks')
        ax.plot(modFirst,reinfFirst,'ro',label='first blocks')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_xlabel('model regressor weight',fontsize=12)
        ax.set_ylabel('reinforcment regressor weight',fontsize=12)
        # ax.set_title(ho)
        ax.legend()
        plt.tight_layout()
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        mod,reinf = [[np.nanmean(np.concatenate([modelWeights[ho][i][wNames.index(reg)] for i in np.where(mouseIds==mid)[0]])) for mid in np.unique(mouseIds)] for reg in ('model','reinforcement')]
        ax.plot(mod,reinf,'ko')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlabel('model weight',fontsize=12)
        ax.set_ylabel('reinforcment weight',fontsize=12)
        # ax.set_title(ho)
        plt.tight_layout()
        

for ho in holdOut:
    reg = [r for r in regressors if r!=ho and r not in ho]
    wNames = sorted(reg)
    wCorr = np.full((len(reg),)*2,np.nan)
    for i,ri in enumerate(reg):
        wi = np.array([np.mean(w[wNames.index(ri)]) for w in modelWeights[ho]])
        for j,rj in enumerate(reg):
            if i!=j:
                wj = np.array([np.mean(w[wNames.index(rj)]) for w in modelWeights[ho]])
                ind = (~np.isnan(wi)) & (~np.isnan(wj))
                wCorr[i,j] = np.corrcoef(wi[ind],wj[ind])[0,1]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    cmax = 1 #np.nanmax(np.absolute(wCorr))
    im = ax.imshow(wCorr,clim=[-cmax,cmax],cmap='bwr')
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(len(wNames)))
    ax.set_yticks(np.arange(len(wNames)))
    ax.set_xlim([-0.5,len(wNames)-0.5])
    ax.set_ylim([len(wNames)-0.5,-0.5])
    ax.set_xticklabels(wNames)
    ax.set_yticklabels(wNames)
    ax.set_title(ho)
    cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
    plt.tight_layout()
        
        
        


    
for i in range(len(exps)):
    w = modelWeights['none'][i]
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
    lickProbModel= 1-cvProbNoLick['none'][i]
    fig = plt.figure(figsize=(8,8))
    ylim = [-0.05,1.05]
    ax = fig.add_subplot(1,1,1)
    for j,(stim,clr) in enumerate(zip(('vis1','vis2','sound1','sound2'),'rmbc')):
        stimInd = sessionStim[i][:lickProbModel.size] == stim
        blockStart = 0
        smoothedLickProb = []
        smoothedLickProbModel = []
        for blockEnd in np.cumsum(sessionBlockTrials[i]):
            if j==0:
                ax.plot([blockEnd+0.5]*2,ylim,'k')
            blockInd = slice(blockStart,blockEnd)
            trialInd = stimInd[blockInd]
            smoothedLickProb.append(gaussian_filter(y[i].astype(float)[:lickProbModel.size][blockInd][trialInd],smoothSigma))
            smoothedLickProbModel.append(gaussian_filter(lickProbModel[blockInd][trialInd],smoothSigma))
            blockStart = blockEnd
        trials = np.where(stimInd)[0]+1
        ax.plot(trials,np.concatenate(smoothedLickProb),color=clr,label=stim+' (mouse, smoothed)')
        ax.plot(trials,np.concatenate(smoothedLickProbModel),'--',color=clr,label=stim+' (model, smoothed)')
        ax.plot(trials,lickProbModel[stimInd],'o',ms=2,mec=clr,mfc='none',label=stim+' (model)')
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



## glm-hmm
num_states = 3        # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 4         # input dimensions

# list of ntrials x nregressors array for each session
inputs = [np.stack([x[reg][i] for reg in regressors],axis=-1) for i in range(len(y))]

choices = [a[:,None].astype(int) for a in y]


glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=num_categories), transitions="standard")
fit_ll = glmhmm.fit(choices, inputs, method="em", num_iters=200, tolerance=10**-4)


weights = glmhmm.observations.params

transitionMatrix = np.exp(glmhmm.transitions.log_Ps)

posteriorProb = [glmhmm.expected_states(d,inpt)[0]for d,inpt in zip(choices,inputs)]

trialsPerSession = [len(a) for a in choices]
latents,testChoices = glmhmm.sample(trialsPerSession[0],input=inputs[0])
testChoices = testChoices.flatten()











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


















