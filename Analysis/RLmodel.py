#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:47:48 2023

@author: samgale
"""

import numpy as np
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42



def choice(q,tau):
    p = np.exp(q/tau)
    p /= p.sum()
    return np.random.choice(len(p),p=p)


def fitModel(fitParamRanges,fixedParams):
    fit = scipy.optimize.brute(calcModelError,fitParamRanges,args=fixedParams,finish=None)
    return fit


def calcModelError(paramsToFit,*fixedParams):
    tauContext,tauAction,alphaContext,alphaAction,penalty = paramsToFit
    exps,useContext = fixedParams
    actualResponse = [obj.trialResponse[~obj.catchTrials] for obj in exps]
    modelResponse = runModel(exps,tauContext,tauAction,alphaContext,alphaAction,penalty,useContext)
    modelError = np.sum((np.concatenate(modelResponse) - np.concatenate(actualResponse))**2)
    return modelError


def runModel(exps,tauContext,tauAction,alphaContext,alphaAction,penalty,useContext=True):
    contextNames = ('vis','sound')
    stimNames = ('vis1','vis2','sound1','sound2')
    
    response = []
    for obj in exps:
        response.append([])
        
        Qcontext = np.zeros(2)
        Qaction = np.zeros((2,4,2))
        Qaction[0,0,1] = 1
        Qaction[0,1:,1] = -1
        if useContext:
            Qaction[1,2,1] = 1
            Qaction[1,[0,1,3],1] = -1
        else:
            Qaction[0,2,1] = 1
        
        for trial,(stim,rewStim,autoRew) in enumerate(zip(obj.trialStim,obj.rewardedStim,obj.autoRewarded)):
            if stim == 'catch':
                continue

            if useContext:
                if trial == 0:
                    context = 0 if 'vis' in stim else 1
                else:
                    context = choice(Qcontext,tauContext)
            else:
                context = 0
            state = stimNames.index(stim)
            action = 1 if trial == 0 or autoRew else choice(Qaction[context,state],tauAction)
            
            if action:
                outcome = 1 if stim==rewStim else penalty
                Qaction[context,state,action] += alphaAction * (outcome - Qaction[context,state,action])
                
                if useContext:
                    if (contextNames[context] in stim and outcome==1) or (contextNames[context] not in stim and outcome < 1):
                        detectedContext = [-1,-1]
                        detectedContext[context] = 1
                    else:
                        detectedContext = [1,1]
                        detectedContext[context] = -1
                    Qcontext += alphaContext * (detectedContext - Qcontext)
            
            response[-1].append(action)
    
    return response
            




tauActionRange = slice(0.05,1,0.1)
alphaActionRange = slice(0.05,1,0.1)
penaltyRange = slice(0,1,1)


modelParams = [[],[]]
modelResponse = [[],[]]
for i,useContext in enumerate((False,True)):
    tauContextRange = slice(0.05,1,0.1) if useContext else slice(0,1,1)
    alphaContextRange = slice(0.05,1,0.1) if useContext else slice(0,1,1)
    fitParamRanges = (tauContextRange,tauActionRange,alphaContextRange,alphaActionRange,penaltyRange)
    for j,exps in enumerate(expsByMouse):
        modelParams[i].append([])
        modelResponse[i].append([])
        for k,testExp in enumerate(exps):
            print(i,j,k)
            trainExps = [obj for obj in exps if obj is not testExp]
            fixedParams = (trainExps,useContext)
            fit = fitModel(fitParamRanges,fixedParams)
            modelParams[i][-1].append(fit)
            tauContext,tauAction,alphaContext,alphaAction,penalty = fit
            modelResponse[i][-1].append(runModel([testExp],tauContext,tauAction,alphaContext,alphaAction,penalty,useContext)[0])



stimNames = ('vis1','vis2','sound1','sound2')

fig = plt.figure()
postTrials = 15
x = np.arange(postTrials)+1
a = 0
for lbl in ('mouse','q learn','context'):
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
        ax = fig.add_subplot(3,2,a+1)
        a += 1
        for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
            y = []
            for i,exps in enumerate(expsByMouse):
                for j,obj in enumerate(exps):
                    if lbl == 'q learn':
                        resp = np.array(modelResponse[0][i][j])
                    elif lbl == 'context':
                        resp = np.array(modelResponse[1][i][j])
                    else:
                        resp = obj.trialResponse[~obj.catchTrials]
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if rewStim==rewardStim:
                            r = resp[(obj.trialBlock[~obj.catchTrials]==blockInd+1) & (obj.trialStim[~obj.catchTrials]==stim) & (~obj.autoRewarded[~obj.catchTrials])]
                            k = min(postTrials,r.size)
                            y.append(np.full(postTrials,np.nan))
                            y[-1][:k] = r[:k]
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y)/(len(y)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stim)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials after block switch cue trials')
        ax.set_ylabel('Response rate')
        ax.legend(loc='lower right')
        ax.set_title(lbl+' '+blockLabel+' (n='+str(len(resp))+')')
plt.tight_layout()












