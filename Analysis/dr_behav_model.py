# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:55:19 2022

@author: svc_ccg
"""

import copy
import os
import re
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData,sortExps
import sklearn
from sklearn.linear_model import LogisticRegression
import psytrack



baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"

mouseIds = ('638573','638574','638575','638576','638577','638578')

excelPath = os.path.join(baseDir,'DynamicRoutingTraining.xlsx')
sheets = pd.read_excel(excelPath,sheet_name=None)
allMiceDf = sheets['all mice']

mice = []
sessionStartTimes = []
passSession =[]
for mid in mouseIds:
    mouseInd = np.where(allMiceDf['mouse id']==int(mid))[0][0]
    df = sheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']])
    firstMultiModal = np.where(['multimodal' in task for task in df['task version']])[0]
    if len(firstMultiModal)>0:
        sessions[firstMultiModal[0]:] = False
    if sessions.sum() > 0:
        mice.append(str(mid))
        sessionStartTimes.append(list(df['start time'][sessions]))
        passSession.append(np.where(df['pass'][sessions])[0][0]-1)
        
expsByMouse = []
for mid,st in zip(mice,sessionStartTimes):
    expsByMouse.append([])
    for t in st:
        f = os.path.join(baseDir,'Data',mid,'DynamicRouting1_' + mid + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5')
        obj = DynRoutData()
        obj.loadBehavData(f)
        expsByMouse[-1].append(obj)

for exps in expsByMouse:
    dprimeSame = np.full((len(exps),6),np.nan)
    dprimeOther = dprimeSame.copy()
    for i,obj in enumerate(exps):
        dprimeSame[i] = obj.dprimeSameModal
        dprimeOther[i] = obj.dprimeOtherModalGo


nTrialsPrev = 15
regressors = ('reinforcement','attention','persistence')
regressorColors = ('k',)
for r,cm in zip(regressors,(plt.cm.autumn,plt.cm.winter,plt.cm.summer)):
    regressorColors += tuple(cm(np.linspace(0,1,nTrialsPrev))[:,:3])

nMice = len(expsByMouse)
nExps = [len(exps) for exps in expsByMouse]
trialsPerSession = [[] for _ in range(nMice)]
trialsPerBlock = copy.deepcopy(trialsPerSession)
trialStim = copy.deepcopy(trialsPerSession)
trialRewardStim = copy.deepcopy(trialsPerSession)
X = [{r: [] for r in regressors} for _ in range(nMice)]
Y = copy.deepcopy(trialsPerSession)
for m,exps in enumerate(expsByMouse):
    for obj in exps:
        trials = ~obj.catchTrials & ~obj.autoRewarded & (obj.trialBlock>1)
        trialInd = np.where(trials)[0]
        nTrials = trials.sum()
        for r in regressors:
            X[m][r].append(np.zeros((nTrials,nTrialsPrev)))
            for n in range(1,nTrialsPrev+1):
                for trial,stim in enumerate(obj.trialStim[trials]):
                    if r in ('reinforcement','persistence'):
                        sameStim = obj.trialStim[:trialInd[trial]] == stim
                        if sameStim.sum()>n:  
                            if r=='reinforcement':
                                if obj.trialResponse[:trialInd[trial]][sameStim][-n]:
                                    X[m][r][-1][trial,n-1] = 1 if obj.trialRewarded[:trialInd[trial]][sameStim][-n] else -1
                            elif r=='persistence':
                                X[m][r][-1][trial,n-1] = obj.trialResponse[:trialInd[trial]][sameStim][-n]
                    elif r=='attention':
                        notCatch = obj.trialStim[:trialInd[trial]] != 'catch'
                        if notCatch.sum()>n:
                            if obj.trialRewarded[:trialInd[trial]][notCatch][-n]:
                                sameModal = any(s in stim and s in obj.trialStim[:trialInd[trial]][notCatch][-n] for s in ('vis','sound'))
                                X[m][r][-1][trial,n-1] = 1 if sameModal else -1
        Y[m].append(obj.trialResponse[trials].astype(float))
        trialsPerSession[m].append(nTrials)
        trialsPerBlock[m].append([np.sum(obj.trialBlock[trials]==block) for block in np.unique(obj.trialBlock[trials])])
        trialStim[m].append(obj.trialStim[trials])
        trialRewardStim[m].append(obj.rewardedStim[trials])



holdOut = ['none']+regressors
accuracy = {h: [[] for m in range(nMice)] for h in holdOut}
balancedAccuracy = copy.deepcopy(accuracy)
prediction = copy.deepcopy(accuracy)
predictionProb = copy.deepcopy(accuracy)
confidence = copy.deepcopy(accuracy)
featureWeights = copy.deepcopy(accuracy)
for h in holdOut:
    for m in range(nMice):
        for i in range(nExps[m]):
            firstTrial = 0
            for blockTrials in trialsPerBlock[m][i]:
                x = np.concatenate([X[m][r][i][firstTrial:firstTrial+blockTrials] for r in regressors if r!=h],axis=1)
                y = Y[m][i][firstTrial:firstTrial+blockTrials]
                firstTrial += blockTrials
                model = LogisticRegression(fit_intercept=True,max_iter=1e3)
                model.fit(x,y)
                accuracy[h][m].append(model.score(x,y))
                
                featureWeights[h][m].append(model.coef_.flatten())
                




stimNames = ('vis1','vis2','sound1','sound2')
postTrials = 15
x = np.arange(postTrials)+1  
for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded block','sound rewarded block')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for stim,clr,ls,lbl in zip(stimNames,'ggmm',('-','--','-','--'),('visual go','visual nogo','auditory go','auditory nogo')):
        resp = []
        for m in range(nMice):
            for i in range(nExps[m]):
                if i>=5:#passSession[m]:
                    continue
                firstTrial = 0
                for blockTrials in trialsPerBlock[m][i]:
                    trials = slice(firstTrial,firstTrial+blockTrials)
                    firstTrial += blockTrials
                    if trialRewardStim[m][i][trials][0]==rewardStim:
                        lick = Y[m][i][trials][trialStim[m][i][trials]==stim][:postTrials]
                        resp.append(np.full(postTrials,np.nan))
                        resp[-1][:lick.size] = lick
        m = np.nanmean(resp,axis=0)
        s = np.nanstd(resp)/(len(resp)**0.5)
        ax.plot(x,m,color=clr,ls=ls,label=lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.legend(loc='lower right')
    ax.set_title(blockLabel)
    plt.tight_layout()


















