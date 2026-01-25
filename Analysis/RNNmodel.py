# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:38:41 2025

@author: svc_ccg
"""

import glob
import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getSessionData
from RNNmodelHPC import getRNNSessions


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"


def boxcar(data,smoothSamples):
    smoothFilter = np.ones(smoothSamples) / smoothSamples
    smoothedData = np.convolve(data,smoothFilter,mode='same')
    smoothedData[:smoothSamples] = smoothedData[smoothSamples]
    smoothedData[-smoothSamples:] = smoothedData[-smoothSamples]
    return smoothedData


#
import shutil
filePaths = glob.glob(os.path.join(baseDir,'Sam','RNNmodel','*.npz'))
for f in filePaths:
    if 'gru' in f:
        shutil.move(f,os.path.join(os.path.dirname(f),'modelComparison',os.path.basename(f)))


#
summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
isStandardRegimen = getIsStandardRegimen(summaryDf)
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])

sessionStartTimes = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = getRNNSessions(mouseId,df)
    sessionStartTimes.append([st.strftime('%Y%m%d_%H%M%S') for st in df.loc[sessions,'start time']])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
dsort = np.sort([len(s) for s in sessionStartTimes])
cumProb = np.array([np.sum(dsort>=i)/dsort.size for i in dsort])
ax.plot(dsort,cumProb,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,dsort[-1]+1])
ax.set_ylim([0,1.01])
ax.set_xlabel('# sessions')
ax.set_ylabel('Cumalative fraction of mice',fontsize=16)
plt.tight_layout()

maxTrainSessions = 20
mouseIds = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = getRNNSessions(mouseId,df)
    if len(sessions) > maxTrainSessions:
        mouseIds.append(mouseId)


# get model data
modelData = {}
filePaths = glob.glob(os.path.join(baseDir,'Sam','RNNmodel','modelComparison','*.npz'))
for f in filePaths:
    fileParts = os.path.splitext(os.path.basename(f))[0].split('_')
    mouseId,sessionDate,sessionTime,hiddenType,nTrainSessions,nHiddenUnits = fileParts
    session = sessionDate+'_'+sessionTime
    nTrainSessions = int(nTrainSessions[:nTrainSessions.find('train')])
    nHiddenUnits = int(nHiddenUnits[:nHiddenUnits.find('hidden')])
    if mouseId not in modelData:
        modelData[mouseId] = {}
    if session not in modelData[mouseId]:
        modelData[mouseId][session] = {'isComplete': False}
    if hiddenType not in modelData[mouseId][session]:
        modelData[mouseId][session][hiddenType] = {}
    if nTrainSessions not in modelData[mouseId][session][hiddenType]:
        modelData[mouseId][session][hiddenType][nTrainSessions] = {}
    d = modelData[mouseId][session][hiddenType][nTrainSessions][nHiddenUnits] = {}
    with np.load(f,allow_pickle=True) as data:
        for key in data.keys():
            d[key] = data[key]

# check for sessions with complete data
hiddenTypes = ('gru',)
nTrainSessions = np.array([4,8,12,16,20])
nHiddenUnits = np.array([2,4,8,16,32])             
mouseIds = []
for mouseId in modelData:
    for session in modelData[mouseId]:
        d = modelData[mouseId][session]
        for hiddenType in hiddenTypes:
            if np.all(np.isin(nTrainSessions,list(d[hiddenType].keys()))):
                if np.all([np.all(np.isin(nHiddenUnits,list(d[hiddenType][key].keys()))) for key in d[hiddenType].keys()]):
                    d['isComplete'] = True

sessionsImported = sum([len(modelData[mouseId]) for mouseId in modelData])
nSessions = sum([modelData[mouseId][session]['isComplete'] for mouseId in modelData for session in modelData[mouseId]])
        
# get session data
sessionData = {mouseId: {} for mouseId in modelData}
for mouseId in modelData:
    for session in modelData[mouseId]:
        if modelData[mouseId][session]['isComplete']:
            sessionData[mouseId][session] = getSessionData(mouseId,session,lightLoad=True)
        
    
#              
for mouseId in modelData:
    for session in modelData[mouseId]:
        for hiddenType in hiddenTypes:
            fig = plt.figure(figsize=(12,10))
            gs = matplotlib.gridspec.GridSpec(nHiddenUnits.size,nTrainSessions.size)
            for i,nh in enumerate(nHiddenUnits[::-1]):
                for j,nt in enumerate(nTrainSessions):
                    ax = fig.add_subplot(gs[i,j])
                    d = modelData[mouseId][session][hiddenType][nt][nh]
                    ax.plot(boxcar(d['logLossTrain'],21),'r',label='train')
                    ax.plot(boxcar(d['logLossTest'],21),'b',label='test')
                    for side in ('right','top'):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(direction='out')
                    ax.set_xlim([-1000,31000])
                    ax.set_ylim([0,0.8])
                    if i==4 and j==2:
                        ax.set_xlabel('Training iteration')
                    if i==2 and j==0:
                        ax.set_ylabel('-log(likelihood)')
                    if i==0 and j==4:
                        ax.legend()
            plt.tight_layout()
        assert(False)


for mouseId in modelData:
    for session in modelData[mouseId]:
        fig = plt.figure(figsize=(4,8))
        for k,hiddenType in enumerate(hiddenTypes):
            ax = fig.add_subplot(len(hiddenTypes),1,k+1)
            logLoss = np.zeros((nHiddenUnits.size,nTrainSessions.size))
            for i,nh in enumerate(nHiddenUnits[::-1]):
                for j,nt in enumerate(nTrainSessions):
                    testLoss = modelData[mouseId][session][hiddenType][nt][nh]['logLossTest']
                    bestIter = np.nanargmin(testLoss)
                    logLoss[i,j] = np.mean(testLoss[bestIter-10:bestIter+11])
            im = ax.imshow(logLoss,cmap='magma',clim=(0.2,0.5))
            cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
            # cb.set_ticks((0,0.2,0.4,0.6))
            # cb.set_ticklabels((0,0.2,0.4,0.6),fontsize=12)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out')
            ax.set_xticks(np.arange(nTrainSessions.size))
            ax.set_yticks(np.arange(nHiddenUnits.size))
            ax.set_xticklabels(nTrainSessions)
            ax.set_yticklabels(nHiddenUnits[::-1])
            if k==2:
                ax.set_xlabel('# Training Sessions')
            if k==1:
                ax.set_ylabel('# Hidden Units')
            ax.set_title(hiddenType+', -log(likelihood)')
        plt.tight_layout()


#
for hiddenType in hiddenTypes:
    logLoss = np.zeros((nSessions,nHiddenUnits.size,nTrainSessions.size))
    k = 0
    for mouseId in modelData:
        for session in modelData[mouseId]:
            if modelData[mouseId][session]['isComplete']:
                for i,nh in enumerate(nHiddenUnits[::-1]):
                    for j,nt in enumerate(nTrainSessions):
                        testLoss = modelData[mouseId][session][hiddenType][nt][nh]['logLossTest']
                        bestIter = np.nanargmin(testLoss)
                        logLoss[k,i,j] = np.mean(testLoss[bestIter-10:bestIter+10])
                k += 1
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(np.mean(logLoss,axis=0),cmap='magma',clim=(0.2,0.5))
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    # cb.set_ticks((0,0.2,0.4,0.6))
    # cb.set_ticklabels((0,0.2,0.4,0.6),fontsize=12)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out')
    ax.set_xticks(np.arange(nTrainSessions.size))
    ax.set_yticks(np.arange(nHiddenUnits.size))
    ax.set_xticklabels(nTrainSessions)
    ax.set_yticklabels(nHiddenUnits[::-1])
    ax.set_xlabel('# Training Sessions')
    ax.set_ylabel('# Hidden Units')
    ax.set_title('-log(likelihood)')
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    clrs = plt.cm.copper(np.linspace(0,1,5))
    mean = logLoss.mean(axis=0)
    sem = logLoss.std(axis=0) / (nSessions**0.5)
    for ym,ys,clr,lbl in zip(mean,sem,clrs,nHiddenUnits[::-1]):
        ax.plot(nTrainSessions,ym,color=clr,alpha=0.5,label=lbl)
        for x,m,s in zip(nTrainSessions,ym,ys):
            ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out')
    ax.set_xlabel('# Training Sessions')
    ax.set_ylabel('-log(likelihood)')
    ax.legend(title='# Hidden Units')
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for ym,ys,clr,lbl in zip(mean.T,sem.T,clrs[::-1],nTrainSessions):
        ax.plot(nHiddenUnits[::-1],ym,color=clr,alpha=0.5,label=lbl)
        for x,m,s in zip(nHiddenUnits[::-1],ym,ys):
            ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out')
    ax.set_xlabel('# Hidden Units')
    ax.set_ylabel('-log(likelihood)')
    ax.legend(title='# Training Sessions')
    plt.tight_layout()
    
    
#
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for hiddenType in hiddenTypes:
    for src in ('mice','model'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
            y = []
            for mouseId in modelData:
                if any([modelData[mouseId][session]['isComplete'] for session in modelData[mouseId]]):
                    y.append([])
                    for session in modelData[mouseId]:
                        if modelData[mouseId][session]['isComplete']:
                            obj = sessionData[mouseId][session]
                            if src == 'mice':
                                resp = obj.trialResponse
                            else:
                                resp = modelData[mouseId][session][hiddenType][16][16]['simulation'].mean(axis=0)
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if blockInd > 0:
                                    stim = np.setdiff1d(obj.blockStimRewarded,rewStim)[0] if 'unrewarded' in stimLbl else rewStim
                                    if 'non-target' in stimLbl:
                                        stim = stim[:-1]+'2'
                                    trials = (obj.trialStim==stim)
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = resp[(obj.trialBlock==blockInd) & trials]
                                    i = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                    post = resp[(obj.trialBlock==blockInd+1) & trials]
                                    if stim==rewStim:
                                        i = min(postTrials,post.size)
                                        y[-1][-1][preTrials:preTrials+i] = post[:i]
                                    else:
                                        i = min(postTrials-5,post.size)
                                        y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                    y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
            ax.plot(x[preTrials:],m[preTrials:],color=clr,ls=ls)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        ax.set_xticks([-5,-1,5,9,14,19])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials after block switch',fontsize=16)
        ax.set_ylabel('Response rate',fontsize=16)
        ax.set_title(src)
        #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
        plt.tight_layout()






