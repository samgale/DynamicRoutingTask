#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:47:48 2023

@author: samgale
"""

import numpy as np
import matplotlib


def softmax(q,tau):
    p = np.exp(q / tau)
    p /= p.sum()
    return p


def softmaxWithBias(q,tau,bias,norm=True):
    p = np.exp((q + bias) / tau)
    p /= p + 1
    if norm:
        low = softmaxWithBias(-1,tau,bias,norm=False)
        high = softmaxWithBias(1,tau,bias,norm=False)
        offset = softmaxWithBias(-1,tau,0,norm=False)
        p -= low
        p *= (1 - 2*offset) / (high - low)
        p += offset
    return p


def runModel(obj,contextMode,visConfidence,audConfidence,alphaContext,tauAction,biasAction,alphaAction,penalty,nIters=10):
    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    
    pContext = np.zeros((nIters,obj.nTrials,2),dtype=float) + 0.5
    
    qAction = np.zeros((nIters,obj.nTrials,2,len(stimNames)),dtype=float) + penalty
    if contextMode == 'no context':
        qAction[:,:,:,[0,2]] = 1
    else:
        qAction[:,:,0,0] = 1
        qAction[:,:,1,2] = 1
    
    response = np.zeros((nIters,obj.nTrials),dtype=int)
    
    for i in range(nIters):
        for trial,(stim,rewStim,autoRew) in enumerate(zip(obj.trialStim,obj.rewardedStim,obj.autoRewardScheduled)):
            if stim == 'catch':
                action = 0
            else:
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]
                
                if contextMode == 'switch context':
                    if trial == 0:
                        context = modality
                    else:
                        context = 0 if pContext[i,trial,0] > 0.5 else 1
                else:
                    context = 0
                    
                if contextMode == 'weight context':
                    q = np.sum(qAction[i,trial] * pStim[None,:] * pContext[i,trial][:,None])
                else:
                    q = np.sum(qAction[i,trial,context] * pStim)
                pAction = softmaxWithBias(q,tauAction,biasAction)
                
                action = 1 if autoRew else np.random.choice(2,p=[1-pAction,pAction])
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qAction[i,trial+1] = qAction[i,trial]
            
                if action:
                    outcome = 1 if stim==rewStim else penalty
                    err = outcome - q
                    
                    if contextMode != 'no context':
                        if outcome < 1:
                            pContext[i,trial+1,modality] -= alphaContext * pStim[0 if modality==0 else 2] * pContext[i,trial,modality]
                        else:
                            pContext[i,trial+1,modality] += alphaContext * (1 - pContext[i,trial,modality]) 
                        pContext[i,trial+1,1 if modality==0 else 0] = 1 - pContext[i,trial+1,modality]
                    
                    if contextMode == 'weight context':
                        qAction[i,trial+1] += alphaAction * pStim[None,:] * pContext[i,trial][:,None] * err
                    else:
                        qAction[i,trial+1,context] += alphaAction * pStim * err
                    qAction[i,trial+1][qAction[i,trial+1] >1] = 1 
                    qAction[i,trial+1][qAction[i,trial+1] <-1] = -1 
            
            response[i,trial] = action
    
    return response, pContext, qAction

