#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:47:48 2023

@author: samgale
"""

import os, copy
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData



def choice(q,tau):
    p = np.exp(q/tau)
    p /= p.sum()
    return np.random.choice(p.size,p=p)


def fitModel(exps,useContext,fitParamRanges):
    actualResponse = np.concatenate([obj.trialResponse for obj in exps])
    lowestError = 1e6
    for params in itertools.product(*fitParamRanges):
        modelResponse = np.concatenate(runModel(exps,useContext,*params))
        modelError = np.sum((modelResponse - actualResponse)**2)
        if modelError < lowestError:
            lowestError = modelError
            bestParams = params
    return bestParams


def runModel(exps,useContext,tauContext,tauAction,alphaContext,alphaAction):
    contextNames = ('vis','sound')
    stimNames = ('vis1','vis2','sound1','sound2')
    
    response = []
    for obj in exps:
        response.append([])
        
        Qcontext = np.zeros(2)
        
        Qaction = np.zeros((2,4,2),dtype=float)
        Qaction[0,0,1] = 1
        Qaction[0,1:,1] = -1
        if useContext:
            Qaction[1,2,1] = 1
            Qaction[1,[0,1,3],1] = -1
        else:
            Qaction[0,2,1] = 1
        
        for trial,(stim,rewStim,autoRew) in enumerate(zip(obj.trialStim,obj.rewardedStim,obj.autoRewarded)):
            if stim == 'catch':
                response[-1].append(0)
            else:
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
                    outcome = 1 if stim==rewStim else -1
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



# plot relationship bewtween tau and difference in q values
dQ = np.arange(-1,1.01,0.01)
tau = np.arange(0.1,4.1,0.1)
p = np.zeros((dQ.size,tau.size))
for i,d in enumerate(dQ):
    for j,t in enumerate(tau):
        p[i,j] = np.exp(d/t)/(np.exp(d/t)+1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(p,clim=(0,1),cmap='hot',aspect='auto')
ax.set_xticks(np.arange(0,41,10))
ax.set_xtickslabels(np.arange(5))
ax.set_yticks(np.arange(0,201,50))
ax.set_ytickslabels(np.arange(-1,1.1,0.5))


# get data
baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"

excelPath = os.path.join(baseDir,'DynamicRoutingTraining.xlsx')
sheets = pd.read_excel(excelPath,sheet_name=None)
allMiceDf = sheets['all mice']

mouseIds = allMiceDf['mouse id']
passOnly = True

mouseIds = ('638573','638574','638575','638576','638577','638578',
            '649943','653481','656726')
passOnly = False

mice = []
sessionStartTimes = []
passSession =[]
for mid in mouseIds:
    if str(mid) in sheets:
        mouseInd = np.where(allMiceDf['mouse id']==int(mid))[0][0]
        df = sheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        if any('stage 3' in task for task in df['task version']) and not any('stage 4' in task for task in df['task version']):
            sessions[np.where(sessions)[0][0]] = False # skipping first 6-block session when preceded by distractor training
        firstExperimentSession = np.where(['multimodal' in task
                                           or 'contrast'in task
                                           or 'opto' in task
                                           or 'nogo' in task
                                           #or 'NP' in rig 
                                           for task,rig in zip(df['task version'],df['rig name'])])[0]
        if len(firstExperimentSession)>0:
            sessions[firstExperimentSession[0]:] = False
        if sessions.sum() > 0 and df['pass'][sessions].sum() > 0:
            mice.append(str(mid))
            if passOnly:
                sessions[:np.where(sessions & df['pass'])[0][0]-1] = False
                passSession.append(0)
            else:
                passSession.append(np.where(df['pass'][sessions])[0][0]-1)
            sessionStartTimes.append(list(df['start time'][sessions]))
        
expsByMouse = []
for mid,st in zip(mice,sessionStartTimes):
    expsByMouse.append([])
    for t in st:
        f = os.path.join(baseDir,'Data',mid,'DynamicRouting1_' + mid + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5')
        obj = DynRoutData()
        obj.loadBehavData(f)
        expsByMouse[-1].append(obj)
        
nMice = len(expsByMouse)
nExps = [len(exps) for exps in expsByMouse]
            


# fit model
modelParams = {'early': {'no context': [], 'context': []}, 'late': {'no context': [], 'context': []}}
modelResponse = copy.deepcopy(modelParams)
for s,stage in enumerate(('early','late')):
    for i,context in enumerate(('no context','context')):
        useContext = context=='context'
        if useContext:
            tauContextRange = (0.25,0.5,1,2,4)
            alphaContextRange = np.arange(0.05,1,0.15)
        else:
            tauContextRange = (0,)
            alphaContextRange = (0,)
        tauActionRange = (0.25,0.5)
        alphaActionRange = np.concatenate(([0.025],np.arange(0.05,1,0.15)))
        fitParamRanges = (tauContextRange,tauActionRange,alphaContextRange,alphaActionRange)
        for j,exps in enumerate(expsByMouse):
            exps = exps[:5] if stage=='early' else exps[passSession[j]:passSession[j]+5]
            modelParams[stage][context].append([])
            modelResponse[stage][context].append([])
            for k,testExp in enumerate(exps):
                print(s,i,j,k)
                trainExps = [obj for obj in exps if obj is not testExp]
                fitParams = fitModel(trainExps,useContext,fitParamRanges)
                modelParams[stage][context][-1].append(fitParams)
                modelResponse[stage][context][-1].append(runModel([testExp],useContext,*fitParams)[0])


# compare model and mice
stimNames = ('vis1','vis2','sound1','sound2')

for stage in ('early','late'):
    fig = plt.figure(figsize=(8,8))
    postTrials = 15
    x = np.arange(postTrials)+1
    a = 0
    for lbl in ('mouse','no context','context'):
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
            ax = fig.add_subplot(3,2,a+1)
            a += 1
            for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                y = []
                for i,exps in enumerate(expsByMouse):
                    exps = exps[:5] if stage=='early' else exps[passSession[i]:passSession[i]+5]
                    for j,obj in enumerate(exps):
                        if lbl == 'mouse':
                            resp = obj.trialResponse
                        else:
                            resp = np.array(modelResponse[stage][lbl][i][j])
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if rewStim==rewardStim:
                                r = resp[(obj.trialBlock==blockInd+1) & (obj.trialStim==stim) & ~obj.autoRewarded]
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
            ax.set_xlabel('Trials after block switch')
            ax.set_ylabel('Response rate')
            if a==1:
                ax.legend(loc='upper right')
            ax.set_title(lbl+', '+blockLabel+' (n='+str(len(resp))+')')
    plt.tight_layout()












