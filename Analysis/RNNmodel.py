# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:38:41 2025

@author: svc_ccg
"""

import copy
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


# plot how many sessions are available for each mouse
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

maxTrainSessions = 16
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
completeSessions = []
incompleteSessions = []
for mouseId in modelData:
    for session in modelData[mouseId]:
        d = modelData[mouseId][session]
        for hiddenType in hiddenTypes:
            if np.all(np.isin(nTrainSessions,list(d[hiddenType].keys()))):
                if np.all([np.all(np.isin(nHiddenUnits,list(d[hiddenType][key].keys()))) for key in d[hiddenType].keys()]):
                    d['isComplete'] = True
                    completeSessions.append((mouseId,session))
                else:
                    incompleteSessions.append((mouseId,session))
 
# get session data
sessionData = {mouseId: {} for mouseId in modelData}
for mouseId in modelData:
    for session in modelData[mouseId]:
        if modelData[mouseId][session]['isComplete']:
            sessionData[mouseId][session] = getSessionData(mouseId,session,lightLoad=True)
        
    
# plot individual session performance        
for mouseId in modelData:
    for session in modelData[mouseId]:
        for hiddenType in hiddenTypes:
            fig = plt.figure(figsize=(12,10))
            gs = matplotlib.gridspec.GridSpec(nHiddenUnits.size,nTrainSessions.size)
            for i,nh in enumerate(nHiddenUnits[::-1]):
                for j,nt in enumerate(nTrainSessions):
                    ax = fig.add_subplot(gs[i,j])
                    d = modelData[mouseId][session][hiddenType][nt][nh]
                    ax.plot(boxcar(d['logLossTrain'],21),'k',label='train')
                    ax.plot(boxcar(d['logLossTest'],21),'r',label='test')
                    for side in ('right','top'):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(direction='out',labelsize=10)
                    ax.set_xlim([-1000,31000])
                    ax.set_ylim([0,0.8])
                    if i==4 and j==2:
                        ax.set_xlabel('Training iteration',fontsize=12)
                    if i==2 and j==0:
                        ax.set_ylabel('-log(likelihood)',fontsize=12)
                    if i==0 and j==4:
                        ax.legend()
            plt.tight_layout()
        assert(False)

for mouseId in modelData:
    for session in modelData[mouseId]:
        for hiddenType in hiddenTypes:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            logLoss = np.zeros((nHiddenUnits.size,nTrainSessions.size))
            for i,nh in enumerate(nHiddenUnits[::-1]):
                for j,nt in enumerate(nTrainSessions):
                    testLoss = modelData[mouseId][session][hiddenType][nt][nh]['logLossTest']
                    bestIter = np.nanargmin(testLoss)
                    logLoss[i,j] = np.mean(testLoss[bestIter-10:bestIter+11])
            likelihood = np.exp(-logLoss)
            im = ax.imshow(likelihood,cmap='magma')
            cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
            # cb.set_ticks((0,0.2,0.4,0.6))
            # cb.set_ticklabels((0,0.2,0.4,0.6),fontsize=12)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',labelsize=10)
            ax.set_xticks(np.arange(nTrainSessions.size))
            ax.set_yticks(np.arange(nHiddenUnits.size))
            ax.set_xticklabels(nTrainSessions)
            ax.set_yticklabels(nHiddenUnits[::-1])
            ax.set_xlabel('# Training Sessions',fontsize=12)
            ax.set_ylabel('# Hidden Units',fontsize=12)
            ax.set_title('likelihood',fontsize=14)
        plt.tight_layout()
        assert(False)


# average model performance across sessions
nSessions = len(completeSessions)
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
    likelihood = np.exp(-logLoss)
    alim = (0.68,0.78)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(np.mean(likelihood,axis=0),cmap='magma',clim=alim)
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
    ax.set_title('likelihood')
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    clrs = plt.cm.copper(np.linspace(0,1,5))
    mean = likelihood.mean(axis=0)
    sem = likelihood.std(axis=0) / (nSessions**0.5)
    for ym,ys,clr,lbl in zip(mean,sem,clrs,nHiddenUnits[::-1]):
        ax.plot(nTrainSessions,ym,color=clr,alpha=0.5,label=lbl)
        for x,m,s in zip(nTrainSessions,ym,ys):
            ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out')
    ax.set_ylim(alim)
    ax.set_xlabel('# Training Sessions')
    ax.set_ylabel('likelihood')
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
    ax.set_ylim(alim)
    ax.set_xlabel('# Hidden Units')
    ax.set_ylabel('likelihood')
    ax.legend(title='# Training Sessions')
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = (0.6,0.9)
    ax.plot(alim,alim,'--',color='0.5')
    ax.plot(np.max(likelihood,axis=(1,2)),likelihood[:,nHiddenUnits==8,nTrainSessions==16],'ko')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out')
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Max likelihood')
    ax.set_ylabel('Likelihood for 8 hidden units and 16 training sessions')
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    n = np.zeros(likelihood.shape[1:])
    for a in likelihood:
        i,j = np.unravel_index(np.argmax(a),a.shape)
        n[i,j] += 1
    n /= len(likelihood)
    im = ax.imshow(n,cmap='magma')
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
    ax.set_title('Fraction of sessions where likelihood is maximal')
    plt.tight_layout()
    

    
# block transition plot
bestNTrainSessions = 16
bestNHiddenUnits = 8
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for hiddenType in hiddenTypes:
    for src in ('mice','model prediction','model simulation'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        for stimLbl,clr,ls in zip(('rewarded target','unrewarded target','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
            y = []
            for mouseId in modelData:
                if any([modelData[mouseId][session]['isComplete'] for session in modelData[mouseId]]):
                    y.append([])
                    for session in modelData[mouseId]:
                        if modelData[mouseId][session]['isComplete']:
                            obj = sessionData[mouseId][session]
                            if src == 'mice':
                                resp = obj.trialResponse
                            elif src == 'model prediction':
                                resp = modelData[mouseId][session][hiddenType][bestNTrainSessions][bestNHiddenUnits]['prediction']
                            elif src == 'model simulation':
                                resp = modelData[mouseId][session][hiddenType][bestNTrainSessions][bestNHiddenUnits]['simulation'].mean(axis=0)
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
        # ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
        plt.tight_layout()


# intra-block resp correlations
def getBlockTrials(obj,block,epoch):
    blockTrials = (obj.trialBlock==block) & ~obj.autoRewardScheduled
    n = blockTrials.sum()
    half = int(n/2)
    startTrial = half if epoch=='last half' else 0
    endTrial = half if epoch=='first half' else n
    return np.where(blockTrials)[0][startTrial:endTrial]


def detrend(r,order=2):
    x = np.arange(r.size)
    return r - np.polyval(np.polyfit(x,r,order),x)


def getCorrelation(r1,r2,rs1,rs2,corrSize=200,detrendOrder=None):
    if detrendOrder is not None:
        r1 = detrend(r1,detrendOrder)
        r2 = detrend(r2,detrendOrder)
        rs1 = rs1.copy()
        rs2 = rs2.copy()
        for z in range(rs1.shape[1]):
            rs1[:,z] = detrend(rs1[:,z],detrendOrder)
            rs2[:,z] = detrend(rs2[:,z],detrendOrder)
    c = np.correlate(r1,r2,'full') / (np.linalg.norm(r1) * np.linalg.norm(r2))   
    cs = np.mean([np.correlate(rs1[:,z],rs2[:,z],'full') / (np.linalg.norm(rs1[:,z]) * np.linalg.norm(rs2[:,z])) for z in range(rs1.shape[1])],axis=0)
    n = c.size // 2 + 1
    corrRaw = np.full(corrSize,np.nan)
    corrRaw[:n] = c[-n:]
    corr = np.full(corrSize,np.nan)
    corr[:n] = (c-cs)[-n:] 
    return corr,corrRaw

epoch = 'full'
stimNames = ('vis1','sound1','vis2','sound2')
autoCorrMat = {src: np.zeros((4,len(modelData),100)) for src in ('mice','model')}
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
corrWithinMat = {src: np.zeros((4,4,len(modelData),200)) for src in ('mice','model')}
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
minTrials = 3
nShuffles = 10
for src in ('mice','model'):
    for m,mouseId in enumerate(modelData):
        if any([modelData[mouseId][session]['isComplete'] for session in modelData[mouseId]]):
            autoCorr = [[] for _ in range(4)]
            autoCorrDetrend = copy.deepcopy(autoCorr)
            corrWithin = [[[] for _ in range(4)] for _ in range(4)]
            corrWithinDetrend = copy.deepcopy(corrWithin)
            for session in modelData[mouseId]:
                if modelData[mouseId][session]['isComplete']:
                    obj = sessionData[mouseId][session]
                    if src=='mice': 
                        trialResponse = [obj.trialResponse]
                    else:    
                        trialResponse = modelData[mouseId][session][hiddenType][bestNTrainSessions][bestNHiddenUnits]['simAction']
                    for tr in trialResponse:
                        resp = np.zeros((4,obj.nTrials))
                        respShuffled = np.zeros((4,obj.nTrials,nShuffles))
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                            for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                                stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                                if len(stimTrials) < minTrials:
                                    continue
                                r = tr[stimTrials].astype(float)
                                r[r<1] = -1
                                resp[i,stimTrials] = r
                                for z in range(nShuffles):
                                    respShuffled[i,stimTrials,z] = np.random.permutation(r)
                        
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                            for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                                stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                                if len(stimTrials) < minTrials:
                                    continue
                                r = resp[i,stimTrials]
                                rs = respShuffled[i,stimTrials]
                                corr,corrRaw = getCorrelation(r,r,rs,rs,100)
                                autoCorr[i].append(corr)
                                corrDetrend,corrRawDetrend = getCorrelation(r,r,rs,rs,100,detrendOrder=2)
                                autoCorrDetrend[i].append(corrDetrend)
                            
                            r = resp[:,blockTrials]
                            rs = respShuffled[:,blockTrials]
                            for i,(r1,rs1) in enumerate(zip(r,rs)):
                                for j,(r2,rs2) in enumerate(zip(r,rs)):
                                    if np.count_nonzero(r1) >= minTrials and np.count_nonzero(r2) >= minTrials:
                                        corr,corrRaw = getCorrelation(r1,r2,rs1,rs2)
                                        corrWithin[i][j].append(corr)
                                        corrDetrend,corrRawDetrend = getCorrelation(r1,r2,rs1,rs2,detrendOrder=2)
                                        corrWithinDetrend[i][j].append(corrDetrend)
                       
            autoCorrMat[src][:,m] = np.nanmean(autoCorr,axis=1)
            autoCorrDetrendMat[src][:,m] = np.nanmean(autoCorrDetrend,axis=1)
                    
            corrWithinMat[src][:,:,m] = np.nanmean(corrWithin,axis=2)
            corrWithinDetrendMat[src][:,:,m] = np.nanmean(corrWithinDetrend,axis=2)

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')


fig = plt.figure(figsize=(12,10))       
gs = matplotlib.gridspec.GridSpec(4,4)
x = np.arange(1,200)
for i,ylbl in enumerate(stimLabels):
    for j,xlbl in enumerate(stimLabels[:4]):
        ax = fig.add_subplot(gs[i,j])
        for lbl,clr in zip(('mice','model'),'kr'):
            mat = corrWithinDetrendMat[lbl][i,j,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,clr,label=lbl)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,20])
        ax.set_ylim([-0.03,0.05])
        if i==3:
            ax.set_xlabel('Lag (trials)',fontsize=14)
        else:
            ax.set_xticklabels([])
        if j==0:
            ax.set_ylabel(ylbl,fontsize=14)
        else:
            ax.set_yticklabels([])
        if i==0:
            ax.set_title(xlbl,fontsize=14)
        if i==0 and j==3:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
plt.tight_layout()



#######
from DynamicRoutingAnalysisUtils import getSessionsToPass
from RNNmodelHPC import CustomRNN, getModelInputAndTarget
import torch

# get data for pooled training
summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
isStandardRegimen = getIsStandardRegimen(summaryDf)
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])

sessionDataByMouse = {phase: [] for phase in ('initial training','after learning')}
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
    sessions = np.where(sessions)[0]
    sessionsToPass = getSessionsToPass(mouseId,df,sessions,stage=5)
    sessionDataByMouse['initial training'].append([getSessionData(mouseId,startTime,lightLoad=True) for startTime in df.loc[sessions[:2],'start time']])
    sessionDataByMouse['after learning'].append([getSessionData(mouseId,startTime,lightLoad=True) for startTime in df.loc[sessions[sessionsToPass:sessionsToPass+2],'start time']])

trainingPhase = 'after learning'
sessionData = (# first session from odd mice, second session from even mice
               [d[0] for d in sessionDataByMouse[trainingPhase][::2]] + [d[1] for d in sessionDataByMouse[trainingPhase][1::2]]
               # second session from odd mice, first session from even mice
               + [d[1] for d in sessionDataByMouse[trainingPhase][::2]] + [d[0] for d in sessionDataByMouse[trainingPhase][1::2]])

testData = sessionData[:len(mice)]
trainData = sessionData[len(mice):]

nTrainSessions = len(trainData)
hiddenType = 'gru'

isFitToMouse = True
isSimulation = not isFitToMouse
inputSize = 6
hiddenSize = 8
outputSize = 1
dropoutProb = 0
learningRate = 0.001 # 0.001
smoothingConstants = (0.9,0.999) # (0.9,0.999)
weightDecay = 0.01 # 0.01
maxTrainIters = 20000
earlyStopThresh = 0.05
earlyStopIters = 500
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
lossFunc = torch.nn.BCELoss()

model = CustomRNN(hiddenType,inputSize,hiddenSize,outputSize,dropoutProb).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=learningRate,betas=smoothingConstants,weight_decay=weightDecay)
logLossTrain = np.full(maxTrainIters,np.nan)
logLossTest = np.full((maxTrainIters,len(testData)),np.nan)
bestIter = 0
trainIndex = 0
for i in range(maxTrainIters):
    print(i)
    session = trainData[trainIndex]
    if trainIndex == nTrainSessions - 1:
        random.shuffle(trainData)
        trainIndex = 0
    else:
        trainIndex += 1
    model.train()
    modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
    modelOutput = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
    loss = lossFunc(modelOutput,targetOutput)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    logLossTrain[i] = loss.item()
    
    if i % 10 == 0:
        model.eval()
        with torch.no_grad():
            for j,session in enumerate(testData):
                modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
                modelOutput = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
                logLossTest[i,j] = lossFunc(modelOutput,targetOutput).item()
            if np.mean(logLossTest[i]) < np.mean(logLossTest[bestIter]):
                bestIter = i
                bestModelStateDict = copy.deepcopy(model.state_dict())
            
        if i > bestIter + earlyStopIters and np.all(np.mean(logLossTest[i-earlyStopIters:i+1]) > np.mean(logLossTest[bestIter]) + earlyStopThresh):
            break

model.load_state_dict(bestModelStateDict)
model.eval()
prediction = []
simulation = []
simAction = []
with torch.no_grad():
    for j,session in enumerate(testData):
        modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
        prediction.append(model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0].cpu().numpy())
        simulation.append([])
        simAction.append([])
        for _ in range(10):
            pAction,action,reward = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation=True)
            simulation[-1].append(pAction.cpu().numpy())
            simAction[-1].append(action.cpu().numpy())


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(len(logLossTrain))
ax.plot(x,logLossTrain,'k',label='train')
ax.plot(x[::10],np.mean(logLossTest,axis=1)[::10],'r',label='test')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out')
ax.set_xlim([-1000,21000])
ax.set_ylim([0,0.8])
ax.set_xlabel('Training iteration')
ax.set_ylabel('-log(likelihood)')
ax.legend()
plt.tight_layout()


preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for src in ('mice','model prediction','model simulation'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for stimLbl,clr,ls in zip(('rewarded target','unrewarded target','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
        y = []
        for sessionInd,obj in enumerate(testData):
            y.append([])
            if src == 'mice':
                resp = obj.trialResponse
            elif src == 'model prediction':
                resp = prediction[sessionInd]
            elif src == 'model simulation':
                resp = np.mean(simulation[sessionInd],axis=0)
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
    # ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
    plt.tight_layout() 
 
 
epoch = 'full'
stimNames = ('vis1','sound1','vis2','sound2')
autoCorrMat = {src: np.zeros((4,1,100)) for src in ('mice','model')}
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
corrWithinMat = {src: np.zeros((4,4,1,200)) for src in ('mice','model')}
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
minTrials = 3
nShuffles = 10
for src in ('mice','model'):
    autoCorr = [[] for _ in range(4)]
    autoCorrDetrend = copy.deepcopy(autoCorr)
    corrWithin = [[[] for _ in range(4)] for _ in range(4)]
    corrWithinDetrend = copy.deepcopy(corrWithin)
    for sessionInd,obj in enumerate(testData):
        if src=='mice': 
            trialResponse = [obj.trialResponse]
        else:    
            trialResponse = simAction[sessionInd]
        for tr in trialResponse:
            resp = np.zeros((4,obj.nTrials))
            respShuffled = np.zeros((4,obj.nTrials,nShuffles))
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                    stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                    if len(stimTrials) < minTrials:
                        continue
                    r = tr[stimTrials].astype(float)
                    r[r<1] = -1
                    resp[i,stimTrials] = r
                    for z in range(nShuffles):
                        respShuffled[i,stimTrials,z] = np.random.permutation(r)
            
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                    stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                    if len(stimTrials) < minTrials:
                        continue
                    r = resp[i,stimTrials]
                    rs = respShuffled[i,stimTrials]
                    corr,corrRaw = getCorrelation(r,r,rs,rs,100)
                    autoCorr[i].append(corr)
                    corrDetrend,corrRawDetrend = getCorrelation(r,r,rs,rs,100,detrendOrder=2)
                    autoCorrDetrend[i].append(corrDetrend)
                
                r = resp[:,blockTrials]
                rs = respShuffled[:,blockTrials]
                for i,(r1,rs1) in enumerate(zip(r,rs)):
                    for j,(r2,rs2) in enumerate(zip(r,rs)):
                        if np.count_nonzero(r1) >= minTrials and np.count_nonzero(r2) >= minTrials:
                            corr,corrRaw = getCorrelation(r1,r2,rs1,rs2)
                            corrWithin[i][j].append(corr)
                            corrDetrend,corrRawDetrend = getCorrelation(r1,r2,rs1,rs2,detrendOrder=2)
                            corrWithinDetrend[i][j].append(corrDetrend)
    
    m = 0
    autoCorrMat[src][:,m] = np.nanmean(autoCorr,axis=1)
    autoCorrDetrendMat[src][:,m] = np.nanmean(autoCorrDetrend,axis=1)
            
    corrWithinMat[src][:,:,m] = np.nanmean(corrWithin,axis=2)
    corrWithinDetrendMat[src][:,:,m] = np.nanmean(corrWithinDetrend,axis=2)

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')


fig = plt.figure(figsize=(12,10))       
gs = matplotlib.gridspec.GridSpec(4,4)
x = np.arange(1,200)
for i,ylbl in enumerate(stimLabels):
    for j,xlbl in enumerate(stimLabels[:4]):
        ax = fig.add_subplot(gs[i,j])
        for lbl,clr in zip(('mice','model'),'kr'):
            mat = corrWithinDetrendMat[lbl][i,j,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,clr,label=lbl)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,20])
        ax.set_ylim([-0.03,0.05])
        if i==3:
            ax.set_xlabel('Lag (trials)',fontsize=14)
        else:
            ax.set_xticklabels([])
        if j==0:
            ax.set_ylabel(ylbl,fontsize=14)
        else:
            ax.set_yticklabels([])
        if i==0:
            ax.set_title(xlbl,fontsize=14)
        if i==0 and j==3:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
plt.tight_layout()

 
 
 
 
 
 
 
 