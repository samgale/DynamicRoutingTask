#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:47:48 2023

@author: samgale
"""

import random
import numpy as np


Qcontext = np.zeros(2)

Qaction = np.zeros((2,4,2))

tauContext = 0.1
tauState = 0.1

alphaContext = 0.1
alphaState = 0.1


contextNames = ('vis','sound')

stimNames = ('vis1','vis2','sound1','sound2')

penalty = 0


for stim,rewStim in zip(obj.trialStim,obj.rewardedStim):
    pContext = np.exp(Qcontext/tauContext)
    pContext /= pContext.sum()
    context = np.random.choice(len(pContext),p=pContext)
    
    state = stimNames.index(stim)
    pAction = np.exp(Qaction[context,state]/tauState)
    pAction /= pAction.sum()
    action = random.choice(len(p),p=p)
    if action:
        outcome = 1 if stim==rewStim else penalty
        Qaction[context,state,action] += alphaState * (outcome - Qaction[context,state,action])
        
        if contextNames[context] in stim and outcome==1 or contextNames[context] not in stim and outcome < 1:
            detectedContext = [-1,-1]
            detectedContext[context] = 1
        else:
            detectedContext = [1,1]
            detectedContext[context] = -1
        Qcontext += alphaContext * (detectedContext - Qcontext)
            
            





















