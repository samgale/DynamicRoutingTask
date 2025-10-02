#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:47:48 2023

@author: samgale
"""

import copy
import datetime
import glob
import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn.metrics
from DynamicRoutingAnalysisUtils import getSessionData, calcDprime
from RLmodelHPC import runModel


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Sam"


##
beta = 2
q = np.arange(0,1,0.01)
bias = np.arange(-10,10,0.1)
Q = beta * (2*q[:,None]-1) + bias[None,:]
p = 1 / (1 + np.exp(-Q))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(p,clim=(0,1),cmap='magma',origin='lower',aspect='auto')
plt.colorbar(im)


## plot relationship bewtween tau and q values
def calcLogisticProb(q,beta,bias,lapse=0):
    return (1 - lapse) / (1 + np.exp(-beta * (2 * (q + bias) - 1)))

q = np.arange(0,1.01,0.01)
beta = np.arange(51)
bias = (0,0.25)
xticks = np.arange(0,q.size+1,int(q.size/4))
yticks = np.arange(0,50,10)
for bi in bias:
    p = np.zeros((beta.size,q.size))
    for i,bt in enumerate(beta):
        p[i] = calcLogisticProb(q,bt,bi,0)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(p,clim=(0,1),cmap='magma',origin='lower',aspect='auto')
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(q[xticks],1))
    ax.set_yticks(yticks)
    ax.set_yticklabels(beta[yticks])
    ax.set_xlabel('Q')
    ax.set_ylabel(r'$\beta$')
    ax.set_title('response probability, bias='+str(bi))
    plt.colorbar(im)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for bt,clr in zip((5,10,20),'rgb'):
    for bi,ls in zip(bias,('-','--')):
        ax.plot(q,calcLogisticProb(q,bt,bi,0),color=clr,ls=ls,label=r'$\beta$='+str(bt)+', bias='+str(bi))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(np.arange(-1,1.1,0.5))
ax.set_yticks(np.arange(0,1.1,0.5))
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('Q',fontsize=14)
ax.set_ylabel('response probability',fontsize=14)
ax.legend()
plt.tight_layout()

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(1,1,1)
for bt,clr in zip((5,),('0.5',)):
    for bi,ls in zip((0,0.15,0.25),('-','--',':')):
        if True:#bi>0:
            ax.plot(q,calcLogisticProb(q,bt,bi,0),color=clr,ls=ls,label='bias='+str(bi))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(np.arange(-1,1.1,0.5))
ax.set_yticks(np.arange(0,1.1,0.5))
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('Expected value',fontsize=14)
ax.set_ylabel('Response probability',fontsize=14)
# ax.legend(loc='lower right',fontsize=12)
plt.tight_layout()


## model simulation with synthetic params
betaAction = 16
biasAction = 0.18
lapseRate = 0
biasAttention = 0
visConfidence = 0.97
audConfidence = 0.92
wContext = 0.6
alphaContext = 0.9
alphaContextNeg = 0.02
tauContext = 110
blockTiming = np.nan
blockTimingShape = np.nan
alphaReinforcement = 0.5
alphaReinforcementNeg = 0.09
tauReinforcement = np.nan
wPerseveration = 0.5
alphaPerseveration = 0.4
tauPerseveration = np.nan
rewardBias = 0.25
rewardBiasTau = 7.5
noRewardBias = np.nan
noRewardBiasTau = np.nan
betaActionOpto = np.nan
biasActionOpto = np.nan

params = (betaAction,biasAction,lapseRate,biasAttention,visConfidence,audConfidence,wContext,alphaContext,alphaContextNeg,tauContext,
          blockTiming,blockTimingShape,alphaReinforcement,alphaReinforcementNeg,tauReinforcement,wPerseveration,alphaPerseveration,tauPerseveration,
          rewardBias,rewardBiasTau,noRewardBias,noRewardBiasTau,
          betaActionOpto,biasActionOpto)

trainingPhase = 'after learning'

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,1,1)
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials)    
ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
d = sessionData[trainingPhase]
for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
    y = []
    for mouse in d:
        y.append([])
        for session in list(d[mouse].keys())[:1]:
            obj = d[mouse][session]
            pContext,qReinforcement,qPerseveration,qReward,qTotal,pAction,action = runModel(obj,*params,useChoiceHistory=False,nReps=1)
            pAction = pAction[0]
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                if blockInd > 0:
                    stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                    trials = (obj.trialStim==stim)# & ~obj.autoRewardScheduled
                    y[-1].append(np.full(preTrials+postTrials,np.nan))
                    pre = pAction[(obj.trialBlock==blockInd) & trials]
                    i = min(preTrials,pre.size)
                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                    post = pAction[(obj.trialBlock==blockInd+1) & trials]
                    if stim==rewStim:
                        i = min(postTrials,post.size)
                        y[-1][-1][preTrials:preTrials+i] = post[:i]
                    else:
                        i = min(postTrials-5,post.size)
                        y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
        y[-1] = np.nanmean(y[-1],axis=0)
    m = np.nanmean(y,axis=0)
    s = np.nanstd(y,axis=0)/(len(y)**0.5)
    ax.plot(x[:preTrials],m[:preTrials],color=clr,label=stimLbl)
    ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
    ax.plot(x[preTrials:],m[preTrials:],color=clr)
    ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks([-5,-1,5,9,14,19])
ax.set_xticklabels([-5,-1,1,5,10,15])
ax.set_yticks([0,0.5,1])
ax.set_xlim([-preTrials-0.5,postTrials-0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Trials of indicated type after block switch',fontsize=14)
ax.set_ylabel('Response rate',fontsize=14)
ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
#ax.set_title(str(len(y))+' mice',fontsize=12)
plt.tight_layout()

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,1,1)
x = np.arange(obj.nTrials) + 1
ax.plot([0,x[-1]+1],[0.5,0.5],'--',color='0.5')
blockStarts = np.where(obj.blockTrial==0)[0]
for i,(b,rewStim) in enumerate(zip(blockStarts,obj.blockStimRewarded)):
    if rewStim == 'vis1':
        w = blockStarts[i+1] - b if i < 5 else obj.nTrials - b
        ax.add_patch(matplotlib.patches.Rectangle([b+1,0],width=w,height=1,facecolor='0.5',edgecolor=None,alpha=0.1,zorder=0))
ax.plot(x,pContext[0][:,0],'k',label='prob vis')
ax.plot(x,qReinforcement[0][:,0],'r',label='reinforcement vis')
ax.plot(x,qReinforcement[0][:,2],'b',label='reinforcement aud')
ax.plot(x,qPerseveration[0][:,0],'m',label='perseveration vis')
ax.plot(x,qPerseveration[0][:,2],'c',label='perseveration aud')
y = 1.05
r = action[0]
for stim,clr in zip(('vis1','sound1'),'rb'):
    for resp in (True,False):
        trials = np.where((obj.trialStim==stim) & (r if resp else ~r))[0] + 1
        ax.vlines(trials,y-0.02,y+0.02,color=clr,alpha=(1 if resp else 0.5))
        y += 0.05
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,x[-1]+1])
ax.set_yticks([0,0.5,1])
# ax.set_ylim([0,1.25])
ax.set_xlabel('Trial',fontsize=12)
ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=12)
plt.tight_layout()


## get fit params from HPC output
fitClusters = False
fitLearningWeights = False
crossValWithinSession = True
outputsPerSession = 1
if fitClusters:
    clustData = np.load(os.path.join(baseDir,'clustData.npy'),allow_pickle=True).item()
    clustIds = (3,4,5,6)
    nClusters = len(clustIds)
    clustColors = ([clr for clr in 'rgkbmcy']+['0.6'])[:nClusters]
    trainingPhases = ('clusters',)
    trainingPhaseColors = 'k'
    outputsPerSession = 4
else:
    trainingPhases = ('initial training','after learning')
    trainingPhaseColors = 'mgrbck'

if fitClusters:
    dirName = ''
    modelTypes = ('ContextRL',)
elif 'opto' in trainingPhases:
    dirName = ''
    modelTypes = ('ContextRL',)
elif fitLearningWeights:
    dirName = 'learning weights'
    modelTypes = ('ContextRL',)
elif crossValWithinSession: 
    dirName = 'learning'
    modelTypes = ('ContextRL',)
else:
    dirName = 'model'
    modelTypes = ('ContextRL',)

modelTypeColors = 'rb'

modelParams = {'visConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
               'audConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
               'wContext': {'bounds': (0,40), 'fixedVal': 0},
               'alphaContext': {'bounds':(0.001,0.999), 'fixedVal': np.nan},
               'alphaContextNeg': {'bounds': (0.001,0.999), 'fixedVal': np.nan},
               'tauContext': {'bounds': (1,300), 'fixedVal': np.nan},
               'blockTiming': {'bounds': (0,1), 'fixedVal': np.nan},
               'blockTimingShape': {'bounds': (0.5,4), 'fixedVal': np.nan},
               'wReinforcement': {'bounds': (0,40), 'fixedVal': 0},
               'alphaReinforcement': {'bounds': (0.001,0.999), 'fixedVal': np.nan},
               'alphaReinforcementNeg': {'bounds': (0.001,0.999), 'fixedVal': np.nan},
               'tauReinforcement': {'bounds': (1,300), 'fixedVal': np.nan},
               'wPerseveration': {'bounds': (0,40), 'fixedVal': 0},
               'alphaPerseveration': {'bounds': (0.001,0.999), 'fixedVal': np.nan},
               'tauPerseveration': {'bounds': (1,600), 'fixedVal': np.nan},
               'wReward': {'bounds': (0,40), 'fixedVal': 0},
               'alphaReward': {'bounds': (0.001,0.999), 'fixedVal': np.nan},
               'tauReward': {'bounds': (1,60), 'fixedVal': np.nan},
               'wBias': {'bounds':(-40,40), 'fixedVal': 0},}

if fitClusters or fitLearningWeights:
    for prm in ('wContext','wReinforcement','wPerseveration','wReward','wBias'):
        for i in range(len((clustIds if fitClusters else trainingPhases))):
            if i > 0:
                modelParams[prm+str(i)] = modelParams[prm]
        
modelParamNames = list(modelParams.keys())
nModelParams = modelParamNames.index('wBias')+1

paramNames = {}
nParams = {}
fixedParamNames = {}
fixedParamLabels = {}
lossParamNames = {}
for modelType in modelTypes:
    paramNames[modelType] = ('visConfidence','audConfidence','wContext','alphaContext','tauContext','wReinforcement','alphaReinforcement',
                             'wReward','alphaReward','tauReward','wBias')
    fixedParamNames[modelType] = ('Full model',)
    fixedParamLabels[modelType] = ('Full model',)
    lossParamNames[modelType] = ('Full model',)
    if fitClusters:
        if modelType == 'ContextRL':
            nParams[modelType] = (14,11,12,11)
            fixedParamNames[modelType] += ('-wContext','-wReinforcement','-wPerseveration')
            fixedParamLabels[modelType] += ('-wContext','-wReinforcement','-wPerseveration')
    elif 'opto' in trainingPhases:
        pass
    else:
        if modelType == 'BasicRL':
            nParams[modelType] = (11,9,8,8,10,14)
            fixedParamNames[modelType] += ('-wReinforcement','-wPerseveration','-wReward','-wBias','+wContext')
            fixedParamLabels[modelType] += ('-wReinforcement','-wPerseveration','-wReward','-wBias','+wContext')
            lossParamNames[modelType] += ('reinforcement','perseveration','reward')
        elif modelType == 'ContextRL':
            nParams[modelType] = (14,)#11,12,9,11,11,13,13)
            fixedParamNames[modelType] += ('-wContext','-Reinforcement','-wContext+wReinforcement','-wReward','-wBias','-tauContext')
            fixedParamLabels[modelType] += ('-wContext','-Reinforcement','-wContext+wReinforcement','-wReward','-wBias','-tauContext')
            lossParamNames[modelType] += ('context','alphaContext','reinforcement','reward','tauContext')
            # nParams[modelType] = (14,11,12,11,11,13)
            # fixedParamNames[modelType] += ('-wContext','-Reinforcement','-wContext+wReinforcement','-wPerseveration','-wReward')
            # fixedParamLabels[modelType] += ('-wContext','-Reinforcement','-wContext+wReinforcement','-wPerseveration','-wReward')
            # lossParamNames[modelType] += ()


modelTypeParams = {}
modelData = {phase: {} for phase in trainingPhases}
dirPath = os.path.join(baseDir,'RLmodel',dirName)
if trainingPhases[0] == 'opto':
    dirPath = os.path.join(dirPath,'opto')
elif fitClusters:
    dirPath = os.path.join(dirPath,'clusters')
filePaths = glob.glob(os.path.join(dirPath,'*.npz'))
for fileInd,f in enumerate(filePaths):
    print(fileInd)
    fileParts = os.path.splitext(os.path.basename(f))[0].split('_')
    mouseId,sessionDate,sessionTime,trainingPhase = fileParts[:4]
    if outputsPerSession > 1:
        modelType = '_'.join(fileParts[4:-1])
        fixedParamsIndex = int(fileParts[-1])
    else:
        modelType = '_'.join(fileParts[4:])
        fixedParamsIndex = None
    if trainingPhase not in trainingPhases or modelType not in modelTypes:
        continue
    session = sessionDate+'_'+sessionTime
    with np.load(f,allow_pickle=True) as data:
        if 'params' not in data:
            continue
        if trainingPhase == 'cluster weights':
            params = []
            prms = data['params'][0]
            for i,clust in enumerate(clustIds):
                p = prms[:nModelParams].copy()
                if i > 0:
                    for w in ('wContext','wReinforcement','wPerseveration','wReward','wBias'):
                        p[modelParamNames.index(w)] = prms[modelParamNames.index(w+str(i))]
                params.append(p)
        elif fitLearningWeights:
            prms = data['params'][0]
            params = prms[:nModelParams].copy()
            i = trainingPhases.index(trainingPhase)
            if i > 0:
                for w in ('wContext','wReinforcement','wPerseveration','wReward','wBias'):
                    params[modelParamNames.index(w)] = prms[modelParamNames.index(w+str(i))]
        elif crossValWithinSession:
            params = data['params'].mean(axis=(0,1))
        else:
            params = data['params']
        if crossValWithinSession:
            logLossTrain = data['logLossTrain'].mean()
            logLossTest = data['logLossTest'].mean()
        else:
            logLossTrain = data['logLossTrain']
            if 'logLossTest' in data:
                logLossTest = data['logLossTest']
            else:
                logLossTest = None
        termMessage = data['terminationMessage']
        if 'trainSessions' in data:
            trainSessions = data['trainSessions']
        else:
            trainSessions = None
        if modelType not in modelTypeParams:
            modelTypeParams[modelType] = {key: val for key,val in data.items() if key not in ('params','logLossTrain','logLossTest','terminationMessage','trainSessions')}
            if 'optoLabel' in modelTypeParams[modelType] and len(modelTypeParams[modelType]['optoLabel'].shape)==0:
                modelTypeParams[modelType]['optoLabel'] = None
    d = modelData[trainingPhase]
    if mouseId not in d:
        d[mouseId] = {}
    if session not in d[mouseId]:
        d[mouseId][session] = {}
    if modelType not in d[mouseId][session]:
        if outputsPerSession > 1:
            d[mouseId][session][modelType] = {key: [None for _ in range(outputsPerSession)] for key in ('params','logLossTrain','logLossTest','terminationMessage','trainSessions')}
        else:
            d[mouseId][session][modelType] = {'params': params, 'logLossTrain': logLossTrain, 'logLossTest': logLossTest, 'terminationMessage': termMessage, 'trainSessions': trainSessions}
    if outputsPerSession > 1:
        p = d[mouseId][session][modelType]
        p['params'][fixedParamsIndex] = params
        p['logLossTrain'][fixedParamsIndex] = logLossTrain
        p['logLossTest'][fixedParamsIndex] = logLossTest
        p['terminationMessage'][fixedParamsIndex] = termMessage
        p['trainSessions'][fixedParamsIndex] = trainSessions
        

# print fit termination message
# for trainingPhase in trainingPhases:
#     for mouse in modelData[trainingPhase]:
#         for session in modelData[trainingPhase][mouse]:
#             for modelType in modelTypes:
#                 print(modelData[trainingPhase][mouse][session][modelType]['terminationMessage'])


## get experiment data and model variables
sessionData = {phase: {} for phase in trainingPhases}
nSim = 10
for trainingPhase in trainingPhases:
    print(trainingPhase)
    d = modelData[trainingPhase]
    for mouse in d:
        for session in d[mouse]:
            if mouse not in sessionData[trainingPhase]:
                sessionData[trainingPhase][mouse] = {session: getSessionData(mouse,session,lightLoad=True)}
            elif session not in sessionData[trainingPhase][mouse]:
                sessionData[trainingPhase][mouse][session] = getSessionData(mouse,session,lightLoad=True)
            obj = sessionData[trainingPhase][mouse][session]
            naivePrediction = np.full(obj.nTrials,obj.trialResponse.mean())
            d[mouse][session]['Naive'] = {'logLossTest': sklearn.metrics.log_loss(obj.trialResponse,naivePrediction),
                                          'BIC': 2 * sklearn.metrics.log_loss(obj.trialResponse,naivePrediction,normalize=False)}
            for modelType in modelTypes:
                if modelType not in d[mouse][session]:
                    continue
                s = d[mouse][session][modelType]
                if fitClusters:
                    s['prediction'] = []
                    s['simulation'] = []
                    s['simAction'] = []
                    s['logLossTest'] = [np.full(nClusters,np.nan) for _ in range(len(fixedParamNames[modelType]))]
                    s['BIC'] = [np.full(nClusters,np.nan) for _ in range(len(fixedParamNames[modelType]))]
                    for k,prms in enumerate(s['params']):
                        pAction = np.full(obj.nTrials,np.nan)
                        pSimulate = pAction.copy()
                        simAction = np.full((nSim,obj.nTrials),np.nan)
                        if prms is not None:
                            for i,clust in enumerate(clustIds):
                                clustTrials = clustData['trialCluster'][mouse][session] == clust
                                if clustTrials.sum() > 0 and not np.all(np.isnan(prms[i])):
                                    params = prms[i]
                                    pContext,qReinforcement,qPerseveration,qReward,qTotal,pAct,action = [val[0] for val in runModel(obj,*params,**modelTypeParams[modelType])]
                                    pAction[clustTrials] = pAct[clustTrials]
                                    pSim,simAct = runModel(obj,*params,useChoiceHistory=False,nReps=nSim,**modelTypeParams[modelType])[-2:]
                                    pSim = np.mean(pSim,axis=0)
                                    pSimulate[clustTrials] = pSim[clustTrials]
                                    simAction[:,clustTrials] = simAct[:,clustTrials]
                                    s['logLossTest'][k][i] = sklearn.metrics.log_loss(obj.trialResponse[clustTrials],pAction[clustTrials])
                                    s['BIC'][k][i] = nParams[modelType][k] * np.log(clustTrials.sum()) + 2 * sklearn.metrics.log_loss(obj.trialResponse[clustTrials],pAction[clustTrials],normalize=False)
                        s['prediction'].append(pAction)
                        s['simulation'].append(pSimulate)
                        s['simAction'].append(simAction)
                else:
                    s['pContext'] = []
                    s['qReinforcement'] = []
                    s['qPerseveration'] = []
                    s['qReward'] = []
                    s['qTotal'] = []
                    s['prediction'] = []
                    if not crossValWithinSession:
                        s['logLossTest'] = []
                    s['BIC'] = []
                    s['simulation'] = []
                    s['simAction'] = []
                    s['simPcontext'] = []
                    s['simQreinforcement'] = []
                    s['simQperseveration'] = []
                    s['logLossSimulation'] = []                   
                    for i,params in enumerate(s['params']):
                        pContext,qReinforcement,qPerseveration,qReward,qTotal,pAction,action = [val[0] for val in runModel(obj,*params,**modelTypeParams[modelType])]
                        s['pContext'].append(pContext)
                        s['qReinforcement'].append(qReinforcement)
                        s['qPerseveration'].append(qPerseveration)
                        s['qReward'].append(qReward)
                        s['qTotal'].append(qTotal)
                        s['prediction'].append(pAction)
                        if 'optoLabel' in modelTypeParams[modelType] and modelTypeParams[modelType]['optoLabel'] is not None:
                            trials = np.in1d(obj.trialOptoLabel,('no opto',)+tuple(modelTypeParams[modelType]['optoLabel']))
                        else:
                            trials = np.ones(obj.nTrials,dtype=bool)
                        if not crossValWithinSession:
                            s['logLossTest'].append(sklearn.metrics.log_loss(obj.trialResponse[trials],pAction[trials]))
                        s['BIC'].append(nParams[modelType][i] * np.log(trials.sum()) + 2 * sklearn.metrics.log_loss(obj.trialResponse[trials],pAction[trials],normalize=False))
                        pContext,qReinforcement,qPerseveration,qReward,qTotal,pAction,action = runModel(obj,*params,useChoiceHistory=False,nReps=nSim,**modelTypeParams[modelType])
                        s['simulation'].append(np.mean(pAction,axis=0))
                        s['simAction'].append(action)
                        s['simPcontext'].append(pContext)
                        s['simQreinforcement'].append(qReinforcement)
                        s['simQperseveration'].append(qPerseveration)
                        s['logLossSimulation'].append(np.mean([sklearn.metrics.log_loss(obj.trialResponse,p) for p in pAction]))


## simulate loss-of-function
for trainingPhase in trainingPhases:
    print(trainingPhase)
    d = modelData[trainingPhase]
    for mouse in d:
        for session in d[mouse]:
            for modelType in modelTypes:
                obj = sessionData[trainingPhase][mouse][session]
                s = d[mouse][session][modelType]
                s['simLossParam'] = []
                s['simLossParamAction'] = []    
                s['simLossParamPcontext'] = []   
                for lossParam in lossParamNames[modelType]:
                    params = s['params'][fixedParamNames[modelType].index('Full model')].copy()
                    noAgent = []
                    if lossParam != 'Full model':
                        for prm in (lossParam if isinstance(lossParam,tuple) else (lossParam,)):
                            if prm in ('context','reinforcement','perseveration','reward'):
                                noAgent.append(prm)
                            else:
                                prmInd = list(modelParams.keys()).index(prm)
                                params[prmInd] =  modelParams[prm]['fixedVal']
                    pContext,qReinforcement,qPerseveration,qReward,qTotal,pAction,action = runModel(obj,*params,noAgent=noAgent,useChoiceHistory=False,nReps=nSim,**modelTypeParams[modelType])
                    s['simLossParam'].append(np.mean(pAction,axis=0))
                    s['simLossParamAction'].append(action)
                    s['simLossParamPcontext'].append(pContext)


## compare model prediction and model simulation  
for trainingPhase in trainingPhases:
    d = modelData[trainingPhase]
    for modelType in modelTypes:
        fig = plt.figure(figsize=(12,10))
        nRows = int(np.ceil(len(fixedParamNames[modelType])/2))
        gs = matplotlib.gridspec.GridSpec(nRows,2)
        row = 0
        col = 0
        for fixedParam in fixedParamNames[modelType]:
            ax = fig.add_subplot(gs[row,col])
            if row == nRows - 1:
                row = 0
                col += 1
            else:
                row += 1
            ax.plot([0,1],[0,1],'k--')
            modelInd = fixedParamNames[modelType].index(fixedParam)
            pred = []
            sim = []
            for mouse in d:
                for session in d[mouse]:
                    if modelType in d[mouse][session]:
                        s = d[mouse][session][modelType]
                        pred.append(np.exp(-s['logLossTest'][modelInd]))
                        sim.append(np.exp(-s['logLossSimulation'][modelInd]))
            ax.plot(pred,sim,'o',mec='k',mfc='none',alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=8)
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.set_aspect('equal')
            ax.set_xlabel('Likelihood of model prediction',fontsize=8)
            ax.set_ylabel('Likelihood of model simulation',fontsize=8)
            r = np.corrcoef(pred,sim)[0,1]
            ax.set_title(('' if fixedParam=='Full model' else 'no ')+str(fixedParam)+'\nr^2 = '+str(round(r**2,2)),fontsize=8)
        plt.tight_layout()
        

## plot performance data
performanceData = {trainingPhase: {modelType: {} for modelType in modelTypes} for trainingPhase in trainingPhases}
for trainingPhase in trainingPhases:
    for modelType in modelTypes:
        for fixedParam in ('mice',) + fixedParamNames[modelType]:
            performanceData[trainingPhase][modelType][fixedParam] = {'respFirst': [],'respLast': [],'dprime': []}
            if fixedParam == 'mice':
                d = sessionData[trainingPhase]
            else:
                d = modelData[trainingPhase]
            for mouse in d:
                respFirst = []
                respLast = []
                dprime = []
                for session in d[mouse]:
                    if modelType not in modelData[trainingPhase][mouse][session]:
                        continue
                    obj = sessionData[trainingPhase][mouse][session]
                    if fixedParam == 'mice':
                        resp = obj.trialResponse
                    else:
                        resp = d[mouse][session][modelType]['simulation'][fixedParamNames[modelType].index(fixedParam)]
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        for stim in ('vis1','sound1'):
                            stimTrials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                            trials = stimTrials & (obj.trialBlock==blockInd+1)
                            n = trials.sum()
                            r = resp[trials].mean()
                            if stim == rewStim:
                                hitRate = r
                                hitTrials = n
                            else:
                                falseAlarmRate = r
                                falseAlarmTrials = n
                                if blockInd > 0: 
                                    respFirst.append(resp[trials][0])
                                    respLast.append(resp[stimTrials & (obj.trialBlock==blockInd)][-1])
                        dprime.append(calcDprime(hitRate,falseAlarmRate,hitTrials,falseAlarmTrials))
                performanceData[trainingPhase][modelType][fixedParam]['respFirst'].append(np.mean(respFirst))
                performanceData[trainingPhase][modelType][fixedParam]['respLast'].append(np.mean(respLast))
                performanceData[trainingPhase][modelType][fixedParam]['dprime'].append(np.mean(dprime))


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(len(modelTypes)+1)
for trainingPhase,clr in zip(trainingPhases,'mg'):
    d = [[performanceData[trainingPhase][modelType][prm]['dprime'] for prm in ('mice','Full model')] for modelType in modelTypes]
    d = np.concatenate(d).T
    mean = np.mean(d,axis=0)
    sem = np.std(d,axis=0)/(len(d)**0.5)
    ax.plot(x,mean,'o',mec=clr,mfc=clr,label=trainingPhase)
    for xi,m,s in zip(x,mean,sem):
        ax.plot([xi,xi],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(x)
ax.set_xticklabels(('Mice',)+modelTypes)
ax.set_yticks(np.arange(0,3,0.5))
ax.set_xlim([-0.25,x[-1]+0.25])
ax.set_ylim([0,2])
ax.set_ylabel('Full block cross-modal d\'',fontsize=12)
ax.legend(loc='upper center')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(len(modelTypes)+1)
ax.plot([-1,len(x)+1],[0,0],'k--')
for trainingPhase,clr in zip(trainingPhases,'mg'):
    d = [[[performanceData[trainingPhase][modelType][prm][lbl] for lbl in ('respFirst','respLast')] for prm in ('mice','Full model')] for modelType in modelTypes]
    d = np.concatenate(d).T
    d = d[:,0] - d[:,1]
    mean = np.mean(d,axis=0)
    sem = np.std(d,axis=0)/(len(d)**0.5)
    ax.plot(x,mean,'o',mec=clr,mfc=clr,label=trainingPhase)
    for xi,m,s in zip(x,mean,sem):
        ax.plot([xi,xi],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(x)
ax.set_xticklabels(('Mice',)+modelTypes)
ax.set_yticks(np.arange(-0.4,0.1,0.1))
ax.set_xlim([-0.25,x[-1]+0.25])
ax.set_ylim([-0.4,0.05])
ax.set_ylabel('$\Delta$ Response rate to non-rewarded target\n(first trial - last trial previous block)')
ax.legend(loc='lower center')
plt.tight_layout()

for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(len(fixedParamLabels[modelType])+1)
    for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
        d = performanceData[trainingPhase][modelType]
        d = np.stack([d[lbl]['dprime'] for lbl in d],axis=1)
        mean = np.mean(d,axis=0)
        sem = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(x,mean,'o',mec=clr,mfc=clr,label=trainingPhase)
        for xi,m,s in zip(x,mean,sem):
            ax.plot([xi,xi],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(('Mice',)+fixedParamLabels[modelType])
    ax.set_xlim([-0.25,len(x)+0.25])
    ax.set_ylim([0,2.2])
    ax.set_ylabel('Cross-modal d\'')
    ax.legend()
    plt.tight_layout()
    
for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(len(fixedParamLabels[modelType])+1)
    ax.plot([-1,len(x)+1],[0,0],'k--')
    for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
        d = performanceData[trainingPhase][modelType]
        respFirst = [d[lbl]['respFirst'] for lbl in d]
        respLast = [d[lbl]['respLast'] for lbl in d]
        d = np.array(respFirst) - np.array(respLast)
        d = d.T
        mean = np.mean(d,axis=0)
        sem = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(x,mean,'o',mec=clr,mfc=clr,label=trainingPhase)
        for xi,m,s in zip(x,mean,sem):
            ax.plot([xi,xi],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(('Mice',)+fixedParamLabels[modelType])
    ax.set_xlim([-0.25,len(x)+0.25])
    # ax.set_ylim([0,2.2])
    ax.set_ylabel('$\Delta$ Response rate to non-rewarded target\n(first trial - last trial previous block)')
    ax.legend()
    plt.tight_layout()
    

## plot model likelihood
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(len(modelTypes)+1)
xlim = [-0.25,xticks[-1]+0.25]
for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
    d = modelData[trainingPhase]
    naive = np.array([np.mean([np.exp(-np.array(session['Naive']['logLossTest'])) for session in mouse.values()],axis=0) for mouse in d.values()])
    lh = [np.array([np.mean([np.exp(-np.array(session[modelType]['logLossTest'][0])) for session in mouse.values() if modelType in session],axis=0) for mouse in d.values()]) for modelType in modelTypes]
    lh = np.stack([naive]+lh,axis=1)
    mean = np.mean(lh,axis=0)
    sem = np.std(lh,axis=0)/(len(lh)**0.5)
    x = np.arange(len(mean))
    ax.plot(x,mean,'o',mec=clr,mfc=clr,label=trainingPhase)
    for xi,m,s in zip(x,mean,sem):
        ax.plot([xi,xi],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xticklabels(('Naive model\n(fixed response\nprobability)',)+modelTypes)
ax.set_xlim(xlim)
ax.set_ylim([0.5,0.75])
ax.set_ylabel('Model likelihood',fontsize=12)
ax.legend()
plt.tight_layout()
    
for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    xticks = np.arange(len(fixedParamLabels[modelType]))
    xlim = [-0.25,xticks[-1]+0.25]
    ax.plot(xlim,[0,0],'k--')
    for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
        d = modelData[trainingPhase]
        lh = np.array([np.mean([np.exp(-np.array(session[modelType]['logLossTest'])) for session in mouse.values() if modelType in session],axis=0) for mouse in d.values()])
        lh -= lh[:,0][:,None]
        mean = np.mean(lh,axis=0)
        sem = np.std(lh,axis=0)/(len(lh)**0.5)
        x = np.arange(len(mean))
        ax.plot(x,mean,'o',mec=clr,mfc=clr,label=trainingPhase)
        for xi,m,s in zip(x,mean,sem):
            ax.plot([xi,xi],[m-s,m+s],color=clr,alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(fixedParamLabels[modelType])
    ax.set_xlim(xlim)
    ax.set_ylabel('$\Delta$ model likelihood',fontsize=12)
    ax.set_title(modelType,fontsize=14)
    ax.legend(loc='lower right')
    plt.tight_layout()
    
# clusters
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(len(modelTypes)+1)
xlim = [-0.25,xticks[-1]+0.25]
for clustInd,(clust,clr) in enumerate(zip(clustIds,clustColors)):
    d = modelData[trainingPhase]
    naive = np.array([np.mean([np.exp(-np.array(session['Naive']['logLossTest'])) for session in mouse.values()],axis=0) for mouse in d.values()])
    lh = [np.array([np.nanmean([np.exp(-np.array(session[modelType]['logLossTest'])[0,clustInd]) for session in mouse.values() if modelType in session],axis=0) for mouse in d.values()]) for modelType in modelTypes]
    lh = np.stack([naive]+lh,axis=1)
    mean = np.nanmean(lh,axis=0)
    sem = np.nanstd(lh,axis=0)/(len(lh)**0.5)
    x = np.arange(len(mean))
    ax.plot(x,mean,'o',mec=clr,mfc=clr,label=trainingPhase)
    for xi,m,s in zip(x,mean,sem):
        ax.plot([xi,xi],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xticklabels(('Naive model\n(fixed response\nprobability)',)+modelTypes)
ax.set_xlim(xlim)
ax.set_ylim([0.5,0.75])
ax.set_ylabel('Model likelihood',fontsize=12)
ax.legend()
plt.tight_layout()

for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    xticks = np.arange(len(fixedParamLabels[modelType]))
    xlim = [-0.25,xticks[-1]+0.25]
    ax.plot(xlim,[0,0],'k--')
    for clustInd,(clust,clr) in enumerate(zip(clustIds,clustColors)):
        d = modelData[trainingPhase]
        lh = np.array([np.nanmean([np.exp(-np.array(session[modelType]['logLossTest'])[:,clustInd]) for session in mouse.values() if modelType in session],axis=0) for mouse in d.values()])
        lh -= lh[:,0][:,None]
        mean = np.nanmean(lh,axis=0)
        sem = np.nanstd(lh,axis=0)/(len(lh)**0.5)
        x = np.arange(len(mean))
        ax.plot(x,mean,'o',mec=clr,mfc=clr,label=trainingPhase)
        for xi,m,s in zip(x,mean,sem):
            ax.plot([xi,xi],[m-s,m+s],color=clr,alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(fixedParamLabels[modelType])
    ax.set_xlim(xlim)
    ax.set_ylabel('$\Delta$ model likelihood',fontsize=12)
    ax.set_title(modelType,fontsize=14)
    # ax.legend(loc='lower right')
    plt.tight_layout()


for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    xticks = np.arange(len(fixedParamLabels[modelType]))
    xlim = [-0.25,xticks[-1]+0.25]
    ax.plot(xlim,[0,0],'k--')
    for trainingPhase,clr in zip(trainingPhases,'mg'):
        d = modelData[trainingPhase]
        lh = np.array([np.mean([session[modelType]['BIC'] for session in mouse.values() if modelType in session],axis=0) for mouse in d.values()])
        lh -= lh[:,0][:,None]
        mean = np.mean(lh,axis=0)
        sem = np.std(lh,axis=0)/(len(lh)**0.5)
        x = np.arange(len(mean))
        ax.plot(x,mean,'o',mec=clr,mfc=clr,label=trainingPhase)
        for xi,m,s in zip(x,mean,sem):
            ax.plot([xi,xi],[m-s,m+s],color=clr,alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(fixedParamLabels[modelType])
    ax.set_xlim(xlim)
    ax.set_ylabel('$\Delta$ BIC',fontsize=12)
    ax.set_title(modelType,fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()

for phase in trainingPhases:    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,1],[0,1],'k--')
    lh = [[np.mean([np.exp(-np.array(session[modelType]['logLossTest'][0])) for session in mouse.values() if modelType in session],axis=0) for mouse in modelData[phase].values()] for modelType in modelTypes]
    ax.plot(lh[0],lh[1],'o',mec='k',mfc='none',alpha=0.5,ms=10)
    mx = np.median(lh[0])
    my = np.median(lh[1])
    madx = scipy.stats.median_abs_deviation(lh[0])
    mady = scipy.stats.median_abs_deviation(lh[1])
    ax.plot(mx,my,'ro',ms=10,alpha=0.5)
    ax.plot([mx,mx],[my-mady,my+mady],'r')
    ax.plot([mx-madx,mx+madx],[my,my],'r') 
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_aspect('equal')
    # ax.set_xticks([0,0.5,1])
    # ax.set_yticks([0,0.5,1])
    alim = [0.35,0.85]
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_xlabel('Model likelihood (basic RL))',fontsize=14)
    ax.set_ylabel('Model likelihood (context RL)',fontsize=14)
    ax.set_title(phase,fontsize=16)
    plt.tight_layout()

# plot likelihood of all model types
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
lh = [np.nanmean([np.mean([np.exp(-np.array(session[modelType]['logLossTest'][0])) for session in mouse.values() if modelType in session],axis=0) for mouse in modelData['after learning'].values()]) for modelType in modelTypes]
lh = np.reshape(lh,(5,4)).T
im = ax.imshow(lh,cmap='magma')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out')
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title('model likelihood')
plt.tight_layout() 

modTypes = ('contextRL_initReinforcement','contextRL_stateSpace_initReinforcement')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
lh = [[np.mean([np.exp(-np.array(session[modelType]['logLossTest'][0])) for session in mouse.values() if modelType in session],axis=0) for mouse in modelData['after learning'].values()] for modelType in modTypes]
ax.plot(lh[0],lh[1],'o',mec='k',mfc='none',alpha=0.5,ms=10)
mx = np.median(lh[0])
my = np.median(lh[1])
madx = scipy.stats.median_abs_deviation(lh[0])
mady = scipy.stats.median_abs_deviation(lh[1])
ax.plot(mx,my,'ro',ms=10)
ax.plot([mx,mx],[my-mady,my+mady],'r')
ax.plot([mx-madx,mx+madx],[my,my],'r') 
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_aspect('equal')
# ax.set_xticks([0,0.5,1])
# ax.set_yticks([0,0.5,1])
alim = [0.35,0.85]
ax.set_xlim(alim)
ax.set_ylim(alim)
# ax.set_xlabel('Model likelihood\n(context RL with context modulation)',fontsize=14)
# ax.set_ylabel('Model likelihood\n(context RL with state-space expansion)',fontsize=14)
ax.set_xlabel('Model likelihood\n(context RL with vector prediction errors)',fontsize=14)
ax.set_ylabel('Model likelihood\n(context RL with scalar prediction errors)',fontsize=14)
plt.tight_layout()

phase = 'noAR'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
lh = [[np.mean([np.exp(-np.array(session['contextRL_learningRates']['logLossTest'][fixedParamNames[modelType].index(fixedParam)])) for session in mouse.values() if modelType in session],axis=0) for mouse in modelData[phase].values()] for fixedParam in (('alphaContextNeg','alphaReinforcementNeg'),'Full model')]
ax.plot(lh[0],lh[1],'o',mec='k',mfc='none',alpha=0.5,ms=10)
mx = np.median(lh[0])
my = np.median(lh[1])
madx = scipy.stats.median_abs_deviation(lh[0])
mady = scipy.stats.median_abs_deviation(lh[1])
ax.plot(mx,my,'ro',ms=10)
ax.plot([mx,mx],[my-mady,my+mady],'r')
ax.plot([mx-madx,mx+madx],[my,my],'r') 
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_aspect('equal')
# ax.set_xticks([0,0.5,1])
# ax.set_yticks([0,0.5,1])
alim = [0.35,0.85]
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_xlabel('Model likelihood\n(context RL model)',fontsize=14)
ax.set_ylabel('Model likelihood\n(context RL with asymmetric learning)',fontsize=14)
plt.tight_layout()
            
    
## plot param values
for modelType in modelTypes:
    fig = plt.figure(figsize=(20,10))
    gs = matplotlib.gridspec.GridSpec(len(fixedParamNames[modelType]),len(paramNames[modelType]))
    for i,fixedParam in enumerate(fixedParamNames[modelType]):
        for j,param in enumerate(paramNames[modelType]):
            ax = fig.add_subplot(gs[i,j])
            for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
                d = modelData[trainingPhase]
                if len(d) > 0:
                    prmInd = list(modelParams.keys()).index(param)
                    paramVals = np.array([np.mean([session[modelType]['params'][i][prmInd] for session in mouse.values() if modelType in session]) for mouse in d.values()])
                    if len(np.unique(paramVals)) > 1:
                        dsort = np.sort(paramVals)
                        cumProb = np.array([np.sum(dsort<=s)/dsort.size for s in dsort])
                        ax.plot(dsort,cumProb,color=clr,label=trainingPhase)
                        print(modelType,fixedParam,param,np.median(paramVals))
                    else:
                        ax.plot(paramVals[0],1,'o',mfc=clr,mec=clr)
                        print(modelType,fixedParam,param,paramVals[0])
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=8)
            xlim = modelParams[param]['bounds']
            ax.set_xlim([xlim[0]-0.02,xlim[1]+0.02])
            ax.set_ylim([0,1.01])
            if j>0:
                ax.set_yticklabels([])
            if i<len(fixedParamNames[modelType])-1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(param,fontsize=8)
            if j==0 and i==len(fixedParamNames[modelType])//2:
                ax.set_ylabel('Cum. Prob.',fontsize=10)
            if j==len(paramNames[modelType])//2:
                ax.set_title(str(fixedParam),fontsize=10)
            if i==0 and j==len(paramNames[modelType])-1:
                ax.legend(bbox_to_anchor=(1,1),fontsize=8)
    plt.tight_layout()

# fig = plt.figure(figsize=(8,12))
# wPrms = [prm for prm in paramNames[modelType] if prm[0]=='w']
# for i,fixedParam in enumerate(fixedParamNames[modelType]):   
#     ax = fig.add_subplot(len(fixedParamNames[modelType]),1,i+1)    
#     for x,param in enumerate(wPrms):
#         for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
#             d = modelData[trainingPhase]
#             if len(d) > 0:
#                 prmInd = list(modelParams.keys()).index(param)
#                 paramVals = np.array([np.mean([session[modelType]['params'][i][prmInd] for session in mouse.values() if modelType in session and session[modelType]['params'][i] is not None]) for mouse in d.values()])
#                 m = np.mean(paramVals)
#                 s = np.std(paramVals) / (len(paramVals)**0.5)
#                 ax.plot(x,m,'o',mec=clr,mfc='none')
#                 ax.plot([x,x],[m-s,m+s],color=clr)
#     for side in ('right','top'):
#         ax.spines[side].set_visible(False)
#     ax.tick_params(direction='out',top=False,right=False)
#     ax.set_xticks(np.arange(len(wPrms)))
#     if i==len(fixedParamNames[modelType])-1:
#         ax.set_xticklabels(wPrms)
#     else:
#         ax.set_xticklabels([])
#     ax.set_xlim([-0.5,len(wPrms)-0.5])
#     ax.set_ylim([0,10])
#     ax.set_title(str(fixedParam))
# plt.tight_layout()

fig = plt.figure(figsize=(8,12))
wPrms = [prm for prm in paramNames[modelType] if prm[0]=='w']
x = np.arange(len(wPrms))
for i,fixedParam in enumerate(fixedParamNames[modelType]):   
    ax = fig.add_subplot(len(fixedParamNames[modelType]),1,i+1)  
    for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
        d = modelData[trainingPhase]
        if len(d) > 0:
            prmInd = [list(modelParams.keys()).index(prm) for prm in wPrms]
            paramVals = np.array([np.mean([session[modelType]['params'][i][prmInd] for session in mouse.values() if modelType in session and session[modelType]['params'][i] is not None],axis=0) for mouse in d.values()])
            # paramVals /= paramVals.sum(axis=1)[:,None]
            m = np.mean(paramVals,axis=0)
            s = np.std(paramVals,axis=0) / (len(paramVals)**0.5)
            ax.plot(x,m,'o',mec=clr,mfc='none')
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(len(wPrms)))
    if i==len(fixedParamNames[modelType])-1:
        ax.set_xticklabels(wPrms)
    else:
        ax.set_xticklabels([])
    ax.set_xlim([-0.5,len(wPrms)-0.5])
    # ax.set_ylim([0,0.5])
    ax.set_title(str(fixedParam))
plt.tight_layout()

#
paramVals = []
phaseInd = []
prmInd = [list(modelParams.keys()).index(prm) for prm in paramNames[modelType]]
for i,trainingPhase in enumerate(trainingPhases):
    d = modelData[trainingPhase]
    paramVals.append(np.array([np.mean([session[modelType]['params'][0][prmInd] for session in mouse.values() if modelType in session and session[modelType]['params'][i] is not None],axis=0) for mouse in d.values()]))
    phaseInd.append(np.zeros(len(paramVals[-1]))+i)

pall = []
for i in range(len(paramNames[modelType])):
    pall.append(scipy.stats.friedmanchisquare(*[vals[:,i] for vals in paramVals])[1])

plt.figure()
plt.imshow(np.array(pall)[None,:],clim=(0,0.05))

p = np.zeros((3,len(paramNames[modelType])))    
for i in range(p.shape[0]):
    for j in range(p.shape[1]):
        p[i,j] = scipy.stats.wilcoxon(paramVals[i][:,j],paramVals[i+1][:,j])[1]
    
plt.figure()
plt.imshow(p,clim=(0,0.05))


from sklearn.linear_model import LogisticRegression

def getTrainTestSplits(y,nSplits=5):
    classVals = np.unique(y)
    nSamples = len(y)
    samplesPerClass = [np.sum(y==val) for val in classVals]
    if any(n < nSplits for n in samplesPerClass):
        return None,None
    samplesPerSplit = [round(n/nSplits) for n in samplesPerClass]
    shuffleInd = np.random.permutation(nSamples)
    trainInd = []
    testInd = []
    for k in range(nSplits):
        testInd.append([])
        for val,n in zip(classVals,samplesPerSplit):
            start = k*n
            ind = shuffleInd[y[shuffleInd]==val]
            testInd[-1].extend(ind[start:start+n] if k+1<nSplits else ind[start:])
        trainInd.append(np.setdiff1d(shuffleInd,testInd[-1]))
    return trainInd,testInd


accuracy = []
coef = []
Crange = 10.0**np.arange(-5,6)
for _ in range(10):
    accuracy.append([])
    coef.append([])
    for i in range(3):
        accuracy[-1].append([])
        coef[-1].append([])
        X = np.concatenate(paramVals[i:i+2],axis=0)
        Xstand = X.copy()
        Xstand -= Xstand.mean(axis=0)
        Xstand /= Xstand.std(axis=0)
        y = np.concatenate(phaseInd[i:i+2])
        outerTrain,outerTest = getTrainTestSplits(y)
        for trainInd,testInd in zip(outerTrain,outerTest):
            innerTrain,innerTest = getTrainTestSplits(y[trainInd])
            a = []
            for train,test in zip(innerTrain,innerTest):
                a.append([])
                for C in Crange:
                    model = LogisticRegression(C=C,max_iter=1e3,penalty='l2',solver='liblinear')
                    model.fit(Xstand[trainInd][train],y[trainInd][train])
                    a[-1].append(model.score(X[trainInd][test],y[trainInd][test]))
            Cbest = Crange[np.argmax(np.mean(a,axis=0))]
            model = LogisticRegression(C=Cbest,max_iter=1e3,penalty='l2',solver='liblinear')
            model.fit(Xstand[trainInd],y[trainInd])
            accuracy[-1][-1].append(model.score(Xstand[testInd],y[testInd]))
            coef[-1][-1].append(model.coef_[0])
        
a = np.mean(accuracy,axis=(0,2))
        
c = np.mean(coef,axis=(0,2))

cmax = np.max(np.absolute(c))
plt.imshow(c,cmap='bwr',clim=(-cmax,cmax))

cnorm = c / np.max(np.absolute(c),axis=1)[:,None]
plt.imshow(cnorm,cmap='bwr',clim=(-1,1))


# clusters
for modelType in modelTypes:
    fig = plt.figure(figsize=(20,10))
    gs = matplotlib.gridspec.GridSpec(len(fixedParamNames[modelType]),len(paramNames[modelType]))
    for i,fixedParam in enumerate(fixedParamNames[modelType]):
        for j,param in enumerate(paramNames[modelType]):
            ax = fig.add_subplot(gs[i,j])
            for clustInd,(clust,clr) in enumerate(zip(clustIds,clustColors)):
                d = modelData[trainingPhase]
                if len(d) > 0:
                    prmInd = list(modelParams.keys()).index(param)
                    paramVals = np.array([np.mean([session[modelType]['params'][i][clustInd][prmInd] for session in mouse.values() if modelType in session and session[modelType]['params'][i] is not None and not np.all(np.isnan(session[modelType]['params'][i][clustInd]))]) for mouse in d.values()])
                    paramVals = paramVals[~np.isnan(paramVals)]
                    if len(paramVals) > 0:
                        if len(np.unique(paramVals)) > 1:
                            dsort = np.sort(paramVals)
                            cumProb = np.array([np.sum(dsort<=s)/dsort.size for s in dsort])
                            ax.plot(dsort,cumProb,color=clr,label=trainingPhase)
                            print(modelType,fixedParam,param,np.median(paramVals))
                        else:
                            ax.plot(paramVals[0],1,'o',mfc=clr,mec=clr)
                            print(modelType,fixedParam,param,paramVals[0])
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=8)
            xlim = modelParams[param]['bounds']
            ax.set_xlim([xlim[0]-0.02,xlim[1]+0.02])
            ax.set_ylim([0,1.01])
            if j>0:
                ax.set_yticklabels([])
            if i<len(fixedParamNames[modelType])-1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(param,fontsize=8)
            if j==0 and i==len(fixedParamNames[modelType])//2:
                ax.set_ylabel('Cum. Prob.',fontsize=10)
            if j==len(paramNames[modelType])//2:
                ax.set_title(str(fixedParam),fontsize=10)
            if i==0 and j==len(paramNames[modelType])-1:
                ax.legend(bbox_to_anchor=(1,1),fontsize=8)
    plt.tight_layout()

fig = plt.figure(figsize=(8,12))
wPrms = [prm for prm in paramNames[modelType] if prm[0]=='w']
for i,fixedParam in enumerate(fixedParamNames[modelType]):   
    ax = fig.add_subplot(len(fixedParamNames[modelType]),1,i+1)    
    for x,param in enumerate(wPrms):
        for clustInd,(clust,clr) in enumerate(zip(clustIds,clustColors)):
            d = modelData[trainingPhase]
            if len(d) > 0:
                prmInd = list(modelParams.keys()).index(param)
                paramVals = np.array([np.mean([session[modelType]['params'][i][clustInd][prmInd] for session in mouse.values() if modelType in session and session[modelType]['params'][i] is not None]) for mouse in d.values()])
                m = np.nanmean(paramVals)
                s = np.nanstd(paramVals) / (len(paramVals)**0.5)
                ax.plot(x,m,'o',mec=clr,mfc='none')
                ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(len(wPrms)))
    if i==len(fixedParamNames[modelType])-1:
        ax.set_xticklabels(wPrms)
    else:
        ax.set_xticklabels([])
    ax.set_xlim([-0.5,len(wPrms)-0.5])
    ax.set_ylim([0,20])
    ax.set_title(str(fixedParam))
plt.tight_layout()


# reinforcement vs perseveration for for full model
alim = (0,30)
for phase in trainingPhases:
    for modelType in modelTypes:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(alim,alim,'k--')
        wr,wp = [np.array([np.mean([session[modelType]['params'][0,list(modelParams.keys()).index(param)] for session in mouse.values() if modelType in session]) for mouse in modelData[phase].values()]) for param in ('wReinforcement','wPerseveration')]
        ax.plot(wr,wp,'o',mec='k',mfc='none',alpha=0.5)
        mx = np.median(wr)
        my = np.median(wp)
        madx = scipy.stats.median_abs_deviation(wr)
        mady = scipy.stats.median_abs_deviation(wp)
        ax.plot(mx,my,'ro',ms=10)
        ax.plot([mx,mx],[my-mady,my+mady],'r')
        ax.plot([mx-madx,mx+madx],[my,my],'r')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_aspect('equal')
        ax.set_xlabel('wReinforcement',fontsize=14)
        ax.set_ylabel('wPerseveration',fontsize=14)
        ax.set_title('median = '+str(round(np.median(wr),1))+', '+str(round(np.median(wp),1)),fontsize=10)
        plt.tight_layout()

# plot tauReinforcment and tauPerseveration for each model type
fig = plt.figure(figsize=(12,10))
gs = matplotlib.gridspec.GridSpec(4,5)
alim = (0,10000)
row = 0
col = 0
for modelType in modelTypes:
    ax = fig.add_subplot(gs[row,col])
    if row == 3:
        row = 0
        col += 1
    else:
        row += 1
    ax.plot(alim,alim,'k--')
    tauR,tauP = [np.array([np.mean([session[modelType]['params'][0,list(modelParams.keys()).index(param)] for session in mouse.values() if modelType in session]) for mouse in modelData['after learning'].values()]) for param in ('tauReinforcement','tauPerseveration')]
    ax.plot(tauP,tauR,'o',mec='k',mfc='none',alpha=0.5)
    mx = np.median(tauP)
    my = np.median(tauR)
    madx = scipy.stats.median_abs_deviation(tauP)
    mady = scipy.stats.median_abs_deviation(tauR)
    ax.plot(mx,my,'ro',ms=10)
    ax.plot([mx,mx],[my-mady,my+mady],'r')
    ax.plot([mx-madx,mx+madx],[my,my],'r')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_yticks(np.arange(0,10001,2500))
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    # ax.set_title(str(round(np.median(tauR)))+', '+str(round(np.median(tauP))),fontsize=8)
plt.tight_layout()

modTypes = ('contextRL_initReinforcement','contextRL_stateSpace_initReinforcement')
fig = plt.figure(figsize=(12,10))
gs = matplotlib.gridspec.GridSpec(4,4)
row = 0
col = 0
for param in paramNames[modTypes[0]]:
    ax = fig.add_subplot(gs[row,col])
    if row == 3:
        row = 0
        col += 1
    else:
        row += 1
    ax.plot((0,10000),(0,10000),'k--',alpha=0.5)
    paramVals = [np.array([np.mean([session[modelType]['params'][0,list(modelParams.keys()).index(param)] for session in mouse.values() if modelType in session]) for mouse in modelData['after learning'].values()]) for modelType in modTypes]
    ax.plot(paramVals[0],paramVals[1],'o',mec='k',mfc='none')
    mx = np.median(paramVals[0])
    my = np.median(paramVals[1])
    madx = scipy.stats.median_abs_deviation(paramVals[0])
    mady = scipy.stats.median_abs_deviation(paramVals[1])
    ax.plot(mx,my,'ro',ms=10)
    ax.plot([mx,mx],[my-mady,my+mady],'r')
    ax.plot([mx-madx,mx+madx],[my,my],'r')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    amin = np.min(paramVals)
    amax = np.max(paramVals)
    amargin = 0.1 * max(abs(amin),abs(amax))
    alim = (amin-amargin,amax+amargin)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_title(param+'\n median x='+str(round(np.median(paramVals[0]),2))+', y='+str(round(np.median(paramVals[1]),2)),fontsize=8)
plt.tight_layout()

phase = 'noAR'
fig = plt.figure(figsize=(12,10))
gs = matplotlib.gridspec.GridSpec(4,4)
row = 0
col = 0
for param in paramNames['contextRL_learningRates']:
    ax = fig.add_subplot(gs[row,col])
    if row == 3:
        row = 0
        col += 1
    else:
        row += 1
    ax.plot((0,10000),(0,10000),'k--',alpha=0.5)
    paramVals = [np.array([np.mean([session['contextRL_learningRates']['params'][fixedParamNames['contextRL_learningRates'].index(fixedParam),list(modelParams.keys()).index(param)] for session in mouse.values() if modelType in session]) for mouse in modelData[phase].values()]) for fixedParam in (('alphaContextNeg','alphaReinforcementNeg'),'Full model')]
    ax.plot(paramVals[0],paramVals[1],'o',mec='k',mfc='none')
    mx = np.median(paramVals[0])
    my = np.median(paramVals[1])
    madx = scipy.stats.median_abs_deviation(paramVals[0])
    mady = scipy.stats.median_abs_deviation(paramVals[1])
    ax.plot(mx,my,'ro',ms=10)
    ax.plot([mx,mx],[my-mady,my+mady],'r')
    ax.plot([mx-madx,mx+madx],[my,my],'r')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    amin = np.min(paramVals)
    amax = np.max(paramVals)
    amargin = 0.1 * max(abs(amin),abs(amax))
    if not np.isnan(amargin):
        alim = (amin-amargin,amax+amargin)
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_aspect('equal')
    ax.set_title(param+'\n median x='+str(round(np.median(paramVals[0]),2))+', y='+str(round(np.median(paramVals[1]),2)),fontsize=8)
plt.tight_layout()

modelType = 'contextRL_learningRates'
phase = 'noAR'
for prms in (('alphaContext','alphaContextNeg'),('alphaReinforcement','alphaReinforcementNeg')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = (0,1)
    ax.plot(alim,alim,'k--')
    alphaPos,alphaNeg = [np.array([np.mean([session[modelType]['params'][0,list(modelParams.keys()).index(param)] for session in mouse.values() if modelType in session]) for mouse in modelData[phase].values()]) for param in prms]
    ax.plot(alphaPos,alphaNeg,'o',mec='k',mfc='none',alpha=0.5)
    mx = np.median(alphaPos)
    my = np.median(alphaNeg)
    madx = scipy.stats.median_abs_deviation(alphaPos)
    mady = scipy.stats.median_abs_deviation(alphaNeg)
    ax.plot(mx,my,'ro',ms=10)
    ax.plot([mx,mx],[my-mady,my+mady],'r')
    ax.plot([mx-madx,mx+madx],[my,my],'r')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_yticks(np.arange(0,10001,2500))
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel(prms[0])
    ax.set_ylabel(prms[1])
    plt.tight_layout()


## compare model and mice
var = 'simulation'
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for modelType in modelTypes:
    for phase in trainingPhases:
        for fixedParam in ('mice','Full model'):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            if fixedParam == 'mice':
                d = sessionData[phase]
            else:
                d = modelData[phase]
            for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
                y = []
                for mouse in d:
                    y.append([])
                    for session in d[mouse]:
                        if modelType not in modelData[phase][mouse][session]:
                            continue
                        obj = sessionData[phase][mouse][session]
                        if fixedParam == 'mice':
                            resp = obj.trialResponse
                        else:
                            resp = d[mouse][session][modelType][var][fixedParamNames[modelType].index(fixedParam)]
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
            # ax.set_title(str(fixedParam))
            #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
            plt.tight_layout()

# plot each fixed param
var = 'simulation'
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for modelType in modelTypes:
    for phase in trainingPhases:
        fig = plt.figure(figsize=(12,10))
        nRows = int(np.ceil((len(fixedParamNames[modelType])+1)/2))
        gs = matplotlib.gridspec.GridSpec(nRows,2)
        row = 0
        col = 0
        for fixedParam in ('mice',) + fixedParamNames[modelType]:
            ax = fig.add_subplot(gs[row,col])
            ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            if row == nRows - 1:
                row = 0
                col += 1
            else:
                row += 1
            if fixedParam == 'mice':
                d = sessionData[phase]
            else:
                d = modelData[phase]
            for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
                y = []
                for mouse in d:
                    y.append([])
                    for session in d[mouse]:
                        if modelType not in modelData[phase][mouse][session]:
                            continue
                        obj = sessionData[phase][mouse][session]
                        if fixedParam == 'mice':
                            resp = obj.trialResponse
                        else:
                            resp = d[mouse][session][modelType][var][fixedParamNames[modelType].index(fixedParam)]
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
            ax.tick_params(direction='out',top=False,right=False,labelsize=12)
            ax.set_xticks([-5,-1,5,9,14,19])
            ax.set_xticklabels([-5,-1,1,5,10,15])
            ax.set_yticks([0,0.5,1])
            ax.set_xlim([-preTrials-0.5,postTrials-0.5])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Trials after block switch',fontsize=14)
            ax.set_ylabel('Response rate',fontsize=14)
            ax.set_title(str(fixedParam))
            #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
            plt.tight_layout()

# loss of function
var = 'simLossParam'
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for modelType in modelTypes:
    for phase in trainingPhases:
        fig = plt.figure(figsize=(12,10))
        nRows = int(np.ceil((len(lossParamNames[modelType])+1)/2))
        gs = matplotlib.gridspec.GridSpec(nRows,2)
        row = 0
        col = 0
        for lossParam in ('mice',) + lossParamNames[modelType]:
            ax = fig.add_subplot(gs[row,col])
            ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            if row == nRows - 1:
                row = 0
                col += 1
            else:
                row += 1
            if lossParam == 'mice':
                d = sessionData[phase]
            else:
                d = modelData[phase]
            for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
                y = []
                for mouse in d:
                    y.append([])
                    for session in d[mouse]:
                        if modelType not in modelData[phase][mouse][session]:
                            continue
                        obj = sessionData[phase][mouse][session]
                        if lossParam == 'mice':
                            resp = obj.trialResponse
                        else:
                            resp = d[mouse][session][modelType][var][lossParamNames[modelType].index(lossParam)]
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
            ax.tick_params(direction='out',top=False,right=False,labelsize=12)
            ax.set_xticks([-5,-1,5,9,14,19])
            ax.set_xticklabels([-5,-1,1,5,10,15])
            ax.set_yticks([0,0.5,1])
            ax.set_xlim([-preTrials-0.5,postTrials-0.5])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Trials after block switch',fontsize=14)
            ax.set_ylabel('Response rate',fontsize=14)
            ax.set_title(str(lossParam))
            #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
            plt.tight_layout()

            
# plot each fixed param for clusters
var = 'simulation'
phase = 'clusters'
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for modelType in modelTypes:
    for clust in clustIds:
        fig = plt.figure(figsize=(12,10))
        nRows = int(np.ceil((len(fixedParamNames[modelType])+1)/2))
        gs = matplotlib.gridspec.GridSpec(nRows,2)
        row = 0
        col = 0
        for fixedParam in ('mice',) + fixedParamNames[modelType]:
            ax = fig.add_subplot(gs[row,col])
            ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            if row == nRows - 1:
                row = 0
                col += 1
            else:
                row += 1
            if fixedParam == 'mice':
                d = sessionData[phase]
            else:
                d = modelData[phase]
            for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
                y = []
                for mouse in d:
                    y.append([])
                    for session in d[mouse]:
                        if modelType not in modelData[phase][mouse][session]:
                            continue
                        clustTrials = clustData['trialCluster'][mouse][session] == clust
                        if clustTrials.sum() == 0:
                            continue
                        obj = sessionData[phase][mouse][session]
                        if fixedParam == 'mice':
                            resp = obj.trialResponse
                        else:
                            resp = d[mouse][session][modelType][var][fixedParamNames[modelType].index(fixedParam)]
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and np.any(clustTrials[obj.trialBlock==blockInd+1]):
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
                    if len(y[-1]) > 0:
                        y[-1] = np.nanmean(y[-1],axis=0)
                    else:
                        y[-1] = np.full(preTrials+postTrials+1,np.nan)
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
                ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
                ax.plot(x[preTrials:],m[preTrials:],color=clr,ls=ls)
                ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=12)
            ax.set_xticks([-5,-1,5,9,14,19])
            ax.set_xticklabels([-5,-1,1,5,10,15])
            ax.set_yticks([0,0.5,1])
            ax.set_xlim([-preTrials-0.5,postTrials-0.5])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Trials after block switch',fontsize=14)
            ax.set_ylabel('Response rate',fontsize=14)
            ax.set_title(str(fixedParam))
            #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
            plt.tight_layout()

# plot full model for each model type
var = 'simulation'
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for phase in trainingPhases:
    fig = plt.figure(figsize=(12,10))
    nRows = 4
    gs = matplotlib.gridspec.GridSpec(4,5)
    row = -1
    col = 0
    for modelType in modelTypes:
        if row == nRows - 1:
            row = 0
            col += 1
        else:
            row += 1
        ax = fig.add_subplot(gs[row,col])
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        if modelType == 'mice':
            d = sessionData[phase]
        else:
            d = modelData[phase]
        for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
            y = []
            for mouse in d:
                y.append([])
                for session in d[mouse]:
                    obj = sessionData[phase][mouse][session]
                    if modelType == 'mice':
                        resp = obj.trialResponse
                    else:
                        if modelType not in d[mouse][session]:
                            continue
                        resp = d[mouse][session][modelType][var][0]
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
                y[-1] = np.nanmean(y[-1],axis=0) if len(y[-1]) > 0 else np.full(x.size,np.nan)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
            ax.plot(x[preTrials:],m[preTrials:],color=clr,ls=ls)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=16)
        ax.set_xticks([-5,-1,5,9,14,19])
        ax.set_yticks([0,0.5,1])
        if row == nRows - 1 and col == 0:
            ax.set_xticklabels([-5,-1,1,5,10,15])
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.01])
        # ax.set_xlabel('Trials after block switch',fontsize=20)
        # ax.set_ylabel('Response rate',fontsize=20)
        # ax.set_title(modelType)
        #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
        plt.tight_layout()

# both block types
for modelType in modelTypes:
    var = 'simulation'
    stimNames = ('vis1','vis2','sound1','sound2')
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)
    for trainingPhase in trainingPhases:
        fig = plt.figure(figsize=(12,10))
        nRows = int(np.ceil((len(fixedParamNames[modelType])+1)/2))
        gs = matplotlib.gridspec.GridSpec(nRows,4)
        for i,fixedParam in enumerate(('mice',) + fixedParamNames[modelType]):
            if fixedParam == 'mice':
                d = sessionData[trainingPhase]
            else:
                d = modelData[trainingPhase]
            if len(d) == 0:
                continue
            for j,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks'))):
                if i>=nRows:
                    row = i-nRows
                    col = j+2
                else:
                    row,col = i,j
                print(row,col,i,j)
                ax = fig.add_subplot(gs[row,col])
                for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                    y = []
                    for mouse in d:
                        y.append([])
                        for session in d[mouse]:
                            if fixedParam != 'mice' and modelType not in d[mouse][session]:
                                continue
                            obj = sessionData[trainingPhase][mouse][session]
                            if fixedParam == 'mice':
                                resp = obj.trialResponse
                            else:
                                resp = d[mouse][session][modelType][var][fixedParamNames[modelType].index(fixedParam)]
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if rewStim==rewardStim and blockInd > 0:
                                    trials = (obj.trialStim==stim) #& ~obj.autoRewardScheduled
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = resp[(obj.trialBlock==blockInd) & trials]
                                    k = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-k:preTrials] = pre[-k:]
                                    post = resp[(obj.trialBlock==blockInd+1) & trials]
                                    k = min(postTrials,post.size)
                                    y[-1][-1][preTrials+1:preTrials+1+k] = post[:k]
                        if len(y[-1]) > 0:
                            y[-1] = np.nanmean(y[-1],axis=0)
                        else:
                            y[-1] = np.full(preTrials+postTrials+1,np.nan)
                    m = np.nanmean(y,axis=0)
                    s = np.nanstd(y,axis=0)/(len(y)**0.5)
                    ax.plot(x,m,color=clr,ls=ls,label=stim)
                    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks(np.arange(-5,20,5))
                ax.set_yticks([0,0.5,1])
                ax.set_xlim([-preTrials-0.5,postTrials+0.5])
                ax.set_ylim([0,1.01])
                if i==len(fixedParamNames[modelType]):
                    ax.set_xlabel('Trials after block switch')
                if j==0:
                    ax.set_ylabel(('Response\nrate' if fixedParam=='mice' else var))
                if fixedParam=='mice':
                    title = 'mice, '+blockLabel+' (n='+str(len(y))+')'
                elif fixedParam=='Full model':
                    title = fixedParam + '(' + modelType + ')'
                else:
                    title = str(fixedParam)# + '=' + str(fixedVal)
                ax.set_title(title)
                if i==0 and j==1:
                    ax.legend(bbox_to_anchor=(1,1))
        plt.tight_layout()

# first block  
for modelType in modelTypes:        
    preTrials = 0
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)
    a = -1
    for var,yticks,ylim,ylbl in zip(('simulation','expectedValue'),([0,0.5,1],[-1,0,1]),([0,1.01],[-1.01,1.01]),('Response\nrate','Expected\nvalue')):
        if var=='expectedValue':
            continue
        for trainingPhase in trainingPhases:
            fig = plt.figure(figsize=(8,10))
            gs = matplotlib.gridspec.GridSpec(3,2)#len(fixedParamNames[modelType])+1,2)
            for i,(fixedParam,fixedVal) in enumerate(zip(('mice',) + fixedParamNames[modelType],(None,)+fixedParamValues[modelType])):
                if fixedParam == 'mice':
                    d = sessionData[trainingPhase]
                elif fixedParam in ('Full model','alphaReinforcement'):
                    d = modelData[trainingPhase]
                else:
                    continue
                if len(d) == 0:
                    continue
                a += 1
                for j,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks'))):
                    ax = fig.add_subplot(gs[a,j])
                    for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                        y = []
                        for mouse in d:
                            y.append([])
                            for session in d[mouse]:
                                obj = sessionData[trainingPhase][mouse][session]
                                if fixedParam == 'mice':
                                    resp = obj.trialResponse
                                else:
                                    resp = d[mouse][session][modelType][var][fixedParamNames[modelType].index(fixedParam)]
                                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                    if rewStim==rewardStim and blockInd == 0:
                                        trials = (obj.trialStim==stim) #& ~obj.autoRewardScheduled
                                        y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                        pre = resp[(obj.trialBlock==blockInd) & trials]
                                        k = min(preTrials,pre.size)
                                        y[-1][-1][preTrials-k:preTrials] = pre[-k:]
                                        post = resp[(obj.trialBlock==blockInd+1) & trials]
                                        k = min(postTrials,post.size)
                                        y[-1][-1][preTrials+1:preTrials+1+k] = post[:k]
                            y[-1] = np.nanmean(y[-1],axis=0)
                        m = np.nanmean(y,axis=0)
                        s = np.nanstd(y,axis=0)/(len(y)**0.5)
                        ax.plot(x,m,color=clr,ls=ls,label=stim)
                        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                    for side in ('right','top'):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False)
                    ax.set_xticks(np.arange(-5,20,5))
                    ax.set_yticks(([0,0.5,1] if fixedParam=='mice' else yticks))
                    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
                    ax.set_ylim(([0,1.01] if fixedParam=='mice' else ylim))
                    if i==len(fixedParamNames):
                        ax.set_xlabel('Trials after block switch')
                    if j==0:
                        ax.set_ylabel(('Response\nrate' if fixedParam=='mice' else ylbl))
                    if fixedParam=='mice':
                        title = 'mice, '+blockLabel+' (n='+str(len(y))+')'
                    elif fixedParam=='Full model':
                        title = fixedParam
                    else:
                        title = fixedParam+'='+str(fixedVal)
                    ax.set_title(title)
                    if i==0 and j==1:
                        ax.legend(bbox_to_anchor=(1,1))
            plt.tight_layout()
        
         
# noAR by first target and reward type
preTrials = 5
postTrials = 16
x = np.arange(-preTrials,postTrials)  
var = 'simAction'
for modelType in modelTypes:
    for trainingPhase in ('noAR',):
        for firstTrialRewStim,blockLbl in zip((True,False),('rewarded target first','non-rewarded target first')):
            for firstTrialLick,lickLbl in zip((True,False),('lick','no lick')):
                fig = plt.figure(figsize=(8,10))
                nRows = int(np.ceil((len(fixedParamNames[modelType])+1)/2))
                gs = matplotlib.gridspec.GridSpec(nRows,4)
                for n,(fixedParam,fixedVal) in enumerate(zip(('mice',) + fixedParamNames[modelType],(None,)+fixedParamValues[modelType])):
                    if fixedParam == 'mice':
                        d = sessionData[trainingPhase]
                    else:
                        d = modelData[trainingPhase]
                    if len(d) == 0:
                        continue
                    ax = fig.add_subplot(len(fixedParamNames[modelType])+1,1,n+1)
                    ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=1,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
                    for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
                        y = []
                        for mouse in d:
                            y.append([])
                            for session in d[mouse]:
                                if fixedParam != 'mice' and modelType not in d[mouse][session]:
                                    continue
                                obj = sessionData[trainingPhase][mouse][session]
                                if fixedParam == 'mice':
                                    resp = [obj.trialResponse]
                                else:
                                    resp = d[mouse][session][modelType][var][fixedParamNames[modelType].index(fixedParam)]
                                for r in resp:
                                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                        if blockInd > 0:
                                            nonRewStim = np.setdiff1d(obj.blockStimRewarded,rewStim)
                                            blockTrials = obj.trialBlock==blockInd+1
                                            firstRewStim = np.where(blockTrials & (obj.trialStim==rewStim))[0][0]
                                            firstNonRewStim = np.where(blockTrials & (obj.trialStim==nonRewStim))[0][0]
                                            if ((firstTrialRewStim and firstRewStim > firstNonRewStim) or
                                                (not firstTrialRewStim and firstRewStim < firstNonRewStim)):
                                                continue
                                            firstTargetTrial = firstRewStim if firstTrialRewStim else firstNonRewStim
                                            if r[firstTargetTrial] != firstTrialLick:
                                                continue
                                            stim = nonRewStim if 'unrewarded' in stimLbl else rewStim
                                            trials = obj.trialStim==stim
                                            y[-1].append(np.full(preTrials+postTrials,np.nan))
                                            pre = r[(obj.trialBlock==blockInd) & trials]
                                            i = min(preTrials,pre.size)
                                            y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                                            post = r[blockTrials & trials]
                                            if (firstTrialRewStim and stim==rewStim) or (not firstTrialRewStim and stim==nonRewStim):
                                                i = min(postTrials,post.size)
                                                y[-1][-1][preTrials:preTrials+i] = post[:i]
                                            else:
                                                i = min(postTrials-1,post.size)
                                                y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
                            if len(y[-1]) > 0:
                                n += len(y[-1])
                                y[-1] = np.nanmean(y[-1],axis=0)
                            else:
                                y[-1] = np.full(preTrials+postTrials,np.nan)
                        if len(y)>0:
                            m = np.nanmean(y,axis=0)
                            s = np.nanstd(y,axis=0)/(len(y)**0.5)
                            ax.plot(x[:preTrials],m[:preTrials],color=clr,label=stimLbl)
                            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
                            ax.plot(x[preTrials:],m[preTrials:],color=clr)
                            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
                    for side in ('right','top'):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False)
                    ax.set_xticks(np.arange(-5,20,5))
                    ax.set_yticks([0,0.5,1])
                    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
                    ax.set_ylim([0,1.01])
                    if i==len(fixedParamNames[modelType]):
                        ax.set_xlabel('Trials after block switch')
                    if j==0:
                        ax.set_ylabel(('Response\nrate' if fixedParam=='mice' else var))
                    if fixedParam=='mice':
                        title = 'mice, '+blockLabel+' (n='+str(len(y))+')'
                    elif fixedParam=='Full model':
                        title = fixedParam + '(' + modelType + ')'
                    else:
                        title = str(fixedParam) + '=' + str(fixedVal)
                    ax.set_title(title)
                    if i==0 and j==1:
                        ax.legend(bbox_to_anchor=(1,1))
                plt.tight_layout()


# opto
trainingPhase = 'opto'
optoLbl = ('lFC','PFC')
stimNames = ('vis1','vis2','sound1','sound2')
xticks = np.arange(len(stimNames))

for modelType in modelTypes:
    for i,(fixedParam,fixedVal) in enumerate(zip(('mice',) + fixedParamNames[modelType],(None,)+fixedParamValues[modelType])):
        if fixedParam == 'mice':
            d = sessionData[trainingPhase]
        else:
            d = modelData[trainingPhase]
        fig = plt.figure()
        fig.suptitle('mice' if fixedParam=='mice' else modelType+', '+str(fixedParam))
        for i,goStim in enumerate(('vis1','sound1')):
            ax = fig.add_subplot(2,1,i+1)
            for lbl,clr in zip(('no opto',optoLbl),'kb'):
                rr = []
                for mouse in d:
                    n = np.zeros(len(stimNames))
                    resp = n.copy()
                    for session in d[mouse]:
                        obj = sessionData[trainingPhase][mouse][session]
                        if fixedParam == 'mice':
                            r = obj.trialResponse
                        else:
                            r = d[mouse][session][modelType]['simulation'][fixedParamNames[modelType].index(fixedParam)]
                        blockTrials = (obj.rewardedStim==goStim) & ~obj.autoRewardScheduled
                        optoTrials = obj.trialOptoLabel=='no opto' if lbl=='no opto' else np.in1d(obj.trialOptoLabel,lbl)
                        for j,stim in enumerate(stimNames):
                            trials = blockTrials & optoTrials & (obj.trialStim==stim)
                            n[j] += trials.sum()
                            resp[j] += r[trials].sum()
                    rr.append(resp/n)
                mean = np.mean(rr,axis=0)
                sem = np.std(rr,axis=0)/(len(rr)**0.5)
                ax.plot(xticks,mean,color=clr,lw=2,label=lbl)
                for x,m,s in zip(xticks,mean,sem):
                    ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks(xticks)
            if i==1:
                ax.set_xticklabels(stimNames)
            else:
                ax.set_xticklabels([])
            ax.set_xlim([-0.25,len(stimNames)-0.75])
            ax.set_ylim([-0.01,1.01])
            ax.set_ylabel('Response Rate')
            ax.legend(bbox_to_anchor=(1,1),loc='upper left')
        plt.tight_layout()

                   
# pContext example
trainingPhase = 'after learning'
modelType = 'contextRL'
fixedParam = 'Full model'
d = modelData[trainingPhase]
for i,mouse in enumerate(list(d.keys())):
    if i not in (36,):
        continue
    for session in d[mouse].keys():
        obj = sessionData[trainingPhase][mouse][session]
        
        s = d[mouse][session][modelType]
        ind = fixedParamNames[modelType].index(fixedParam)
        pContext = s['simPcontext'][ind]
        qReinforcement = s['simQreinforcement'][ind]
        qPerseveration = s['simQperseveration'][ind]
        action = s['simAction'][ind]
        params = s['params'][ind]
        # print(params[paramNames[modelType].index('alphaReinforcement')])
        
        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(1,1,1)
        x = np.arange(obj.nTrials) + 1
        ax.plot([0,x[-1]+1],[0.5,0.5],'--',color='0.5')
        blockStarts = np.where(obj.blockTrial==0)[0]
        for i,(b,rewStim) in enumerate(zip(blockStarts,obj.blockStimRewarded)):
            if rewStim == 'vis1':
                w = blockStarts[i+1] - b if i < 5 else obj.nTrials - b
                ax.add_patch(matplotlib.patches.Rectangle([b+1,0],width=w,height=1,facecolor='0.5',edgecolor=None,alpha=0.1,zorder=0))
        ax.plot(x,pContext[0][:,0],'k',label='prob vis')
        ax.plot(x,qReinforcement[0][:,0],'r',label='reinforcement vis')
        ax.plot(x,qReinforcement[0][:,2],'b',label='reinforcement aud')
        ax.plot(x,qPerseveration[0][:,0],'m',label='perseveration vis')
        ax.plot(x,qPerseveration[0][:,2],'c',label='perseveration aud')
        y = 1.05
        r = action[0]
        for stim,clr in zip(('vis1','sound1'),'rb'):
            for resp in (True,False):
                trials = np.where((obj.trialStim==stim) & (r if resp else ~r))[0] + 1
                ax.vlines(trials,y-0.02,y+0.02,color=clr,alpha=(1 if resp else 0.5))
                y += 0.05
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,x[-1]+1])
        ax.set_yticks([0,0.5,1])
        # ax.set_ylim([0,1.25])
        ax.set_xlabel('Trial',fontsize=12)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=12)
        plt.tight_layout()
        

# time dependence of effect of prior reward or response
stimType = ('rewarded target','non-rewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
prevTrialTypes = ('response to rewarded target','response to non-rewarded target')
modTypes = ('mice',) + modelTypes
fxdPrms = copy.deepcopy(fixedParamNames)
fxdPrms['mice'] = (None,)
blockEpochs = ('full','first half','last half')
resp = {modelType: {fixedParam: {phase: {epoch: {s: [] for s in stimType} for epoch in blockEpochs} for phase in trainingPhases} for fixedParam in fxdPrms[modelType]} for modelType in modTypes}
respNorm = copy.deepcopy(resp)
trialsSince = {modelType: {fixedParam: {phase: {epoch: {prevTrial: {s: [] for s in stimType} for prevTrial in prevTrialTypes} for epoch in blockEpochs} for phase in trainingPhases} for fixedParam in fxdPrms[modelType]} for modelType in modTypes}
timeSince = copy.deepcopy(trialsSince)
for modelType in modTypes:
    for fixedParam in fxdPrms[modelType]:
        for phase in trainingPhases:
            for epoch in blockEpochs:
                for mouse in modelData[phase]:
                    for i,session in enumerate(modelData[phase][mouse]):
                        obj = sessionData[phase][mouse][session]
                        if modelType=='mice': 
                            trialResponse = [obj.trialResponse]
                        else:
                            if modelType not in modelData[phase][mouse][session]:
                                continue
                            trialResponse = modelData[phase][mouse][session][modelType]['simAction'][fixedParamNames[modelType].index(fixedParam)]
                        b = 0
                        for r in trialResponse:
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                                blockTrials = np.setdiff1d(blockTrials,np.where(obj.catchTrials)[0])
                                rewTrials = np.intersect1d(blockTrials,np.where(obj.trialRewarded)[0])
                                rewTargetTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==rewStim)[0])
                                otherModalTarget = np.setdiff1d(obj.blockStimRewarded,rewStim)[0]
                                nonRewTargetTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==otherModalTarget)[0])
                                targetTrials = np.concatenate((rewTargetTrials,nonRewTargetTrials))
                                nonTargetTrials = np.setdiff1d(blockTrials,targetTrials)
                                for s in stimType:
                                    if i == 0 and b == 0:
                                        resp[modelType][fixedParam][phase][epoch][s].append([])
                                        respNorm[modelType][fixedParam][phase][epoch][s].append([])
                                    if s=='rewarded target':
                                        stim = rewStim
                                    elif s=='non-rewarded target':
                                        stim = otherModalTarget
                                    elif s=='non-target (rewarded modality)':
                                        stim = rewStim[:-1]+'2'
                                    else:
                                        stim = otherModalTarget[:-1]+'2'
                                    stimTrials = obj.trialStim == stim
                                    stimTrials = np.intersect1d(blockTrials,np.where(stimTrials)[0])
                                    if len(stimTrials) < 1:
                                        continue
                                    for prevTrialType,trials in zip(prevTrialTypes,(rewTargetTrials,nonRewTargetTrials)):
                                        if i == 0 and b == 0:
                                            trialsSince[modelType][fixedParam][phase][epoch][prevTrialType][s].append([])
                                            timeSince[modelType][fixedParam][phase][epoch][prevTrialType][s].append([])
                                        respTrials = np.intersect1d(trials,np.where(r)[0])
                                        if len(respTrials) > 0:
                                            prevRespTrial = respTrials[np.searchsorted(respTrials,stimTrials) - 1]
                                            anyTargetTrials = np.array([np.any(np.in1d(obj.trialStim[p+1:s],(rewStim,otherModalTarget))) for s,p in zip(stimTrials,prevRespTrial)])
                                            anyQuiescentViolations = np.array([np.any(obj.trialQuiescentViolations[p+1:s]) for s,p in zip(stimTrials,prevRespTrial)])
                                            notValid = (stimTrials <= respTrials[0]) | (stimTrials > trials[-1]) #| anyTargetTrials #| anyQuiescentViolations
                                            # if len(rewTrials) > 0 and prevTrialType != 'response to rewarded target':
                                            #     prevRewTrial = rewTrials[np.searchsorted(rewTrials,stimTrials) - 1]
                                            #     notValid = notValid | ((stimTrials - prevRewTrial) < 2)
                                            tr = stimTrials - prevRespTrial
                                            tr[notValid] = -1
                                            tm = obj.stimStartTimes[stimTrials] - obj.stimStartTimes[prevRespTrial]
                                            tm[notValid] = np.nan
                                            trialsSince[modelType][fixedParam][phase][epoch][prevTrialType][s][-1].extend(tr)
                                            timeSince[modelType][fixedParam][phase][epoch][prevTrialType][s][-1].extend(tm)
                                        else:
                                            trialsSince[modelType][fixedParam][phase][epoch][prevTrialType][s][-1].extend(np.full(len(stimTrials),np.nan))
                                            timeSince[modelType][fixedParam][phase][epoch][prevTrialType][s][-1].extend(np.full(len(stimTrials),np.nan))
                                    resp[modelType][fixedParam][phase][epoch][s][-1].extend(r[stimTrials])
                                    respNorm[modelType][fixedParam][phase][epoch][s][-1].extend(r[stimTrials] - r[stimTrials].mean())
                                b += 1
            
                for i,prevTrialType in enumerate(prevTrialTypes):
                    for s in stimType:
                        trialsSince[modelType][fixedParam][phase][epoch][prevTrialType][s] = [np.array(a) for a in trialsSince[modelType][fixedParam][phase][epoch][prevTrialType][s]]
                        timeSince[modelType][fixedParam][phase][epoch][prevTrialType][s] = [np.array(a) for a in timeSince[modelType][fixedParam][phase][epoch][prevTrialType][s]]
                        if i==0:
                            resp[modelType][fixedParam][phase][epoch][s] = [np.array(a) for a in resp[modelType][fixedParam][phase][epoch][s]]
                            respNorm[modelType][fixedParam][phase][epoch][s] = [np.array(a) for a in respNorm[modelType][fixedParam][phase][epoch][s]]

timeBins = np.array([0,5,10,15,20,30,40,50,60,80,100])
x = timeBins[:-1] + np.diff(timeBins)/2
epoch = 'full'
for prevTrialType in prevTrialTypes: 
    for modelType in modelTypes:
        for phase in trainingPhases:
            fig = plt.figure(figsize=(12,10))
            fig.suptitle(modelType+', '+phase)
            nRows = int(np.ceil((len(fixedParamNames[modelType])+1)/2))
            gs = matplotlib.gridspec.GridSpec(nRows,2)
            row = 0
            col = 0
            for fixedParam in ('mice',) + fixedParamNames[modelType]:
                ax = fig.add_subplot(gs[row,col])
                if row == nRows - 1:
                    row = 0
                    col += 1
                else:
                    row += 1
                clrs = 'gmgm'
                mt,fp = ('mice',None) if fixedParam=='mice' else (modelType,fixedParam)
                for stim,clr,ls in zip(stimType,clrs,('-','-','--','--')):
                    n = []
                    p = []
                    for d,r in zip(timeSince[mt][fp][phase][epoch][prevTrialType][stim],respNorm[mt][fp][phase][epoch][stim]):
                        n.append(np.full(x.size,np.nan))
                        p.append(np.full(x.size,np.nan))
                        for i,t in enumerate(timeBins[:-1]):
                            j = (d >= t) & (d < timeBins[i+1])
                            n[-1][i] = j.sum()
                            p[-1][i] = r[j].sum() / n[-1][i]
                    m = np.nanmean(p,axis=0)
                    s = np.nanstd(p,axis=0) / (len(p)**0.5)
                    ax.plot(x,m,color=clr,ls=ls,label=stim)
                    ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_yticks(np.arange(-0.5,0.5,0.1))
                ax.set_xlim([0,90])
                ax.set_ylim([-0.1,0.2])
                if row == 0 and col == 1:
                    ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=14)
                if row == 2 and col == 0:
                    ax.set_ylabel('Response rate\n(difference from within-block mean)',fontsize=14)
                ax.set_title(fixedParam)
            plt.tight_layout()

stim = 'non-rewarded target'
for prevTrialType in prevTrialTypes: 
    for modelType in modelTypes:
        for phase in trainingPhases:
            fig = plt.figure(figsize=(12,10))
            fig.suptitle(modelType+', '+phase)
            nRows = int(np.ceil((len(fixedParamNames[modelType])+1)/2))
            gs = matplotlib.gridspec.GridSpec(nRows,2)
            row = 0
            col = 0
            for fixedParam in ('mice',) + fixedParamNames[modelType]:
                ax = fig.add_subplot(gs[row,col])
                if row == nRows - 1:
                    row = 0
                    col += 1
                else:
                    row += 1
                clrs = 'gmgm'
                mt,fp = ('mice',None) if fixedParam=='mice' else (modelType,fixedParam)
                for epoch,clr in zip(blockEpochs,'krb'):
                    n = []
                    p = []
                    for d,r in zip(timeSince[mt][fp][phase][epoch][prevTrialType][stim],resp[mt][fp][phase][epoch][stim]):
                        n.append(np.full(x.size,np.nan))
                        p.append(np.full(x.size,np.nan))
                        for i,t in enumerate(timeBins[:-1]):
                            j = (d >= t) & (d < timeBins[i+1])
                            n[-1][i] = j.sum()
                            p[-1][i] = r[j].sum() / n[-1][i]
                    m = np.nanmean(p,axis=0)
                    s = np.nanstd(p,axis=0) / (len(p)**0.5)
                    ax.plot(x,m,color=clr,label=epoch)
                    ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_xlim([0,90])
                ax.set_ylim([0.3,0.8])
                if row == 0 and col == 1:
                    ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=14)
                if row == 2 and col == 0:
                    ax.set_ylabel('Response rate to '+stim,fontsize=14)
                ax.set_title(fixedParam)
            plt.tight_layout()
                    

## effect of prior reward or response
stimType = ('rewarded target','non-rewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
prevTrialTypes = ('rewarded','response to non-rewarded target','non-response to non-rewarded target','response to same stimulus','non-response to same stimulus')
modTypes = ('mice',) + modelTypes
fxdPrms = copy.deepcopy(fixedParamNames)
fxdPrms['mice'] = (None,)
resp = {modelType: {fixedParam: {phase: {prevTrialType: {stim: [] for stim in stimType} for prevTrialType in prevTrialTypes} for phase in trainingPhases} for fixedParam in fxdPrms[modelType]} for modelType in modTypes}
respMean = copy.deepcopy(resp)
for modelType in modTypes:
    for fixedParam in fxdPrms[modelType]:
        for phase in trainingPhases:
            for prevTrialType in prevTrialTypes:
                for s in stimType:
                    for mouse in modelData[phase]:
                        r = []
                        rm = []
                        for session in modelData[phase][mouse]:
                            obj = sessionData[trainingPhase][mouse][session]
                            if modelType=='mice': 
                                trialResponse = [obj.trialResponse]
                            else:
                                trialResponse = modelData[phase][mouse][session][modelType]['simAction'][fixedParamNames[modelType].index(fixedParam)]
                            for tr in trialResponse:
                                tr = tr.astype(bool)
                                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                    nonRewStim = 'sound1' if rewStim=='vis1' else 'vis1'
                                    if s=='rewarded target':
                                        stim = rewStim
                                    elif s=='non-rewarded target':
                                        stim = nonRewStim
                                    elif s=='non-target (rewarded modality)':
                                        stim = rewStim[:-1]+'2'
                                    else:
                                        stim = nonRewStim[:-1]+'2'
                                    stimTrials = np.where(obj.trialStim==stim)[0]    
                                    blockTrials = np.where(~obj.autoRewardScheduled & (obj.trialBlock==blockInd+1))[0]
                                    trials = np.intersect1d(stimTrials,blockTrials)
                                    if prevTrialType == 'rewarded':
                                        ind = tr & (obj.trialStim == rewStim)
                                    elif prevTrialType == 'response to non-rewarded target':
                                        ind = tr & (obj.trialStim == nonRewStim)
                                    elif prevTrialType == 'non-response to non-rewarded target':
                                        ind = ~tr & (obj.trialStim == nonRewStim)
                                    elif prevTrialType == 'response to same stimulus':
                                        ind = tr & (obj.trialStim == stim)
                                    elif prevTrialType == 'non-response to same stimulus':
                                        ind = ~tr & (obj.trialStim == stim)
                                    r.append(tr[trials][ind[trials-1]])
                                    rm.append(np.mean(tr[trials]))
                        if len(r) > 0:
                            r = np.concatenate(r)
                        resp[modelType][fixedParam][phase][prevTrialType][s].append(np.nanmean(r))
                        respMean[modelType][fixedParam][phase][prevTrialType][s].append(np.nanmean(rm))

alim = (0,1.02)
for modelType in modTypes:
    for fixedParam in fxdPrms[modelType]:
        for phase in trainingPhases:
            for prevTrialType in prevTrialTypes:
                fig = plt.figure(figsize=(7.5,5))
                ax = fig.add_subplot(1,1,1)
                ax.plot(alim,alim,'k--')
                for stim,mec,mfc in zip(stimType,'gmgm',('g','m','none','none')):
                    ax.plot(respMean[modelType][fixedParam][phase][prevTrialType][stim],resp[modelType][fixedParam][phase][prevTrialType][stim],'o',color=mec,mec=mec,mfc=mfc,label=stim)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=10)
                ax.set_xlim(alim)
                ax.set_ylim(alim)
                ax.set_aspect('equal')
                ax.set_xlabel('Response rate'+'\n(within-block mean)',fontsize=14)
                ax.set_ylabel('Response rate'+'\n(previous trial '+prevTrialType+')',fontsize=14)
                ax.legend(loc=('upper left' if 'non-response' in prevTrialType else 'lower right'),fontsize=12)
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

modTypes = ('mice',) + modelTypes
fxdPrms = copy.deepcopy(fixedParamNames)
fxdPrms['mice'] = (None,)
blockEpochs = ('full',) #'first half','last half')
stimNames = ('vis1','sound1','vis2','sound2')
autoCorrMat = {modelType: {fixedParam: {phase: {epoch: np.zeros((4,len(modelData[phase]),100)) for epoch in blockEpochs} for phase in trainingPhases} for fixedParam in fxdPrms[modelType]} for modelType in modTypes}
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
corrWithinMat = {modelType: {fixedParam: {phase:{epoch: np.zeros((4,4,len(modelData[phase]),200)) for epoch in blockEpochs} for phase in trainingPhases} for fixedParam in fxdPrms[modelType]} for modelType in modTypes}
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
minTrials = 3
nShuffles = 10
for modelType in modTypes:
    for fixedParam in fxdPrms[modelType]:
        for phase in trainingPhases:
            for epoch in blockEpochs:
                for m,mouse in enumerate(modelData[phase]):
                    autoCorr = [[] for _ in range(4)]
                    autoCorrDetrend = copy.deepcopy(autoCorr)
                    corrWithin = [[[] for _ in range(4)] for _ in range(4)]
                    corrWithinDetrend = copy.deepcopy(corrWithin)
                    for session in modelData[phase][mouse]:
                        if modelType != 'mice' and modelType not in modelData[phase][mouse][session]:
                            continue
                        if mouse not in sessionData[phase] or session not in sessionData[phase][mouse]:
                            continue
                        obj = sessionData[phase][mouse][session]
                        if modelType=='mice': 
                            trialResponse = [obj.trialResponse]
                        else:
                            ind = fixedParamNames[modelType].index(fixedParam)
                            trialResponse = modelData[phase][mouse][session][modelType]['simAction'][ind]
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
                               
                    autoCorrMat[modelType][fixedParam][phase][epoch][:,m] = np.nanmean(autoCorr,axis=1)
                    autoCorrDetrendMat[modelType][fixedParam][phase][epoch][:,m] = np.nanmean(autoCorrDetrend,axis=1)
                            
                    corrWithinMat[modelType][fixedParam][phase][epoch][:,:,m] = np.nanmean(corrWithin,axis=2)
                    corrWithinDetrendMat[modelType][fixedParam][phase][epoch][:,:,m] = np.nanmean(corrWithinDetrend,axis=2)

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')

modelType = 'MixedAgentRL'        
phase = 'after learning'
for fixedParam in fixedParamNames[modelType]:
    fig = plt.figure(figsize=(5,10))    
    fig.suptitle(fixedParam)         
    gs = matplotlib.gridspec.GridSpec(4,1)
    x = np.arange(1,100)
    for i,lbl in enumerate(stimLabels):
        ax = fig.add_subplot(gs[i])
        for mod,clr in zip(('mice',modelType),'kr'):
            mat = autoCorrDetrendMat[mod][(None if mod=='mice' else fixedParam)][phase]['full'][i,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,color=clr,label=mod)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(0,20,5))
        ax.set_xlim([0,10])
        ax.set_ylim([-0.06,0.2])
        if i==3:
            ax.set_xlabel('Lag (trials of same stimulus)',fontsize=11)
        if i==0:
            ax.set_ylabel('Autocorrelation',fontsize=12)
        ax.set_title(lbl,fontsize=12)
        if i==0 and j==3:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
    plt.tight_layout()
        
fig = plt.figure()           
gs = matplotlib.gridspec.GridSpec(4,1)
x = np.arange(1,100)
ax = fig.add_subplot(1,1,1)
for modelType,clr,lbl in zip(modTypes,'kgm',('mice','vector prediction error','scalar prediction error')):
    mat = autoCorrDetrendMat[modelType][fxdPrms[modelType][0]]['after learning']['full'][1,:,1:]
    m = np.nanmean(mat,axis=0)
    s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(np.arange(0,20,5))
ax.set_xlim([0,10])
# ax.set_ylim([-0.06,0.2])
ax.set_xlabel('Lag (trials of same stimulus)',fontsize=12)
ax.set_ylabel('Autocorrelation',fontsize=12)
ax.legend()
plt.tight_layout()

for modelType in modTypes:
    for fixedParam in fxdPrms[modelType]:
        for i,stim in enumerate(stimLabels):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot([0,0],[0,1],'k--')
            for phase,clr in zip(trainingPhases,'mg'):
                d = autoCorrDetrendMat[modelType][fixedParam][phase]['full'][i,:,1]
                dsort = np.sort(d)
                cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
                ax.plot(dsort,cumProb,color=clr,label=phase)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=12)
            ax.set_xlim([-0.1,0.25])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Autocorrelation of responses',fontsize=14)
            ax.set_ylabel('Cumalative fraction of mice',fontsize=14)
            ax.set_title(stim.replace('\n',' '),fontsize=14)
            plt.legend(loc='lower right')
            plt.tight_layout() 


modelType = 'ContextRL'        
phase = 'after learning'
for fixedParam in fixedParamNames[modelType]:
    fig = plt.figure(figsize=(12,10))
    fig.suptitle(fixedParam)         
    gs = matplotlib.gridspec.GridSpec(4,4)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:4]):
            ax = fig.add_subplot(gs[i,j])
            for mod,clr in zip(('mice',modelType),'kr'):
                mat = corrWithinDetrendMat[mod][(None if mod=='mice' else fixedParam)][phase]['full'][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr,label=mod)
                ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,20])
            ax.set_ylim([-0.025,0.04])
            if i==3:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
            if i==0 and j==3:
                ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
    plt.tight_layout()
                    

# no reward blocks, target stimuli only
for modelType in modelTypes:
    fig = plt.figure(figsize=(8,10))
    gs = matplotlib.gridspec.GridSpec(len(fixedParamNames[modelType])+1,2)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)  
    for i,fixedParam in enumerate(('mice',) + fixedParamNames[modelType]):
        if fixedParam == 'mice':
            d = sessionData['no reward']
        else:
            d = modelData['no reward']
        if len(d) == 0:
            continue
        for j,(blockRewarded,title) in enumerate(zip((True,False),('switch to rewarded block','switch to unrewarded block'))):
            ax = fig.add_subplot(gs[i,j])
            ax.plot([0,0],[0,1],'--',color='0.5')
            for stimLbl,clr in zip(('previously rewarded target stim','other target stim'),'mg'):
                y = []
                for mouse in d:
                    y.append([])
                    for session in d[mouse]:
                        obj = sessionData['no reward'][mouse][session]
                        if fixedParam == 'mice':
                            resp = obj.trialResponse
                        else:
                            resp = d[mouse][session][modelType]['simulation'][fixedParamNames[modelType].index(fixedParam)]
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and ((blockRewarded and rewStim != 'none') or (not blockRewarded and rewStim == 'none')):
                                if blockRewarded:
                                    stim = np.setdiff1d(('vis1','sound1'),rewStim) if 'previously' in stimLbl else rewStim
                                else:
                                    prevRewStim = obj.blockStimRewarded[blockInd-1]
                                    stim = np.setdiff1d(('vis1','sound1'),prevRewStim) if 'other' in stimLbl else prevRewStim
                                trials = (obj.trialStim==stim)
                                y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                pre = resp[(obj.trialBlock==blockInd) & trials]
                                k = min(preTrials,pre.size)
                                y[-1][-1][preTrials-k:preTrials] = pre[-k:]
                                post = resp[(obj.trialBlock==blockInd+1) & trials]
                                k = min(postTrials,post.size)
                                y[-1][-1][preTrials+1:preTrials+1+k] = post[:k]
                    y[-1] = np.nanmean(y[-1],axis=0)
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x,m,color=clr,label=stimLbl)
                ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xticks(np.arange(-20,21,5))
            ax.set_yticks([0,0.5,1])
            ax.set_xlim([-preTrials-0.5,postTrials+0.5])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
            ax.set_ylabel('Response rate',fontsize=12)
            # ax.legend(bbox_to_anchor=(1,1),loc='upper left')
            # ax.set_title(title+' ('+str(len(y))+' mice)',fontsize=12)
    plt.tight_layout()

for modelType in modelTypes:
    for var,ylbl in zip(('pContext','wHabit'),('Context belief','Habit weight')):
        fig = plt.figure(figsize=(10,10))
        gs = matplotlib.gridspec.GridSpec(len(fixedParamNames)+1,2)
        preTrials = 20
        postTrials = 60
        x = np.arange(-preTrials,postTrials+1)  
        for i,(fixedParam,fixedVal) in enumerate(zip(fixedParamNames,fixedParamValues)):
            d = modelData['no reward']
            if len(d) == 0:
                continue
            for j,(blockRewarded,blockLabel) in enumerate(zip((True,False),('switch to rewarded block','switch to unrewarded block'))):
                ax = fig.add_subplot(gs[i,j])
                ax.plot([0,0],[0,1],'--',color='0.5')
                contexts,clrs = (('visual','auditory'),'gm') if var=='pContext' else ((None,),'k')
                for contextInd,(context,clr) in enumerate(zip(contexts,clrs)):
                    y = []
                    for mouse in d:
                        y.append([])
                        for session in d[mouse]:
                            obj = sessionData['no reward'][mouse][session]
                            v = d[mouse][session][modelType][var][fixedParamNames.index(fixedParam)]
                            if var=='pContext':
                                v = v[:,contextInd]
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if blockInd > 0 and ((blockRewarded and rewStim != 'none') or (not blockRewarded and rewStim == 'none')):
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = v[obj.trialBlock==blockInd]
                                    k = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-k:preTrials] = pre[-k:]
                                    post = v[obj.trialBlock==blockInd+1]
                                    k = min(postTrials,post.size)
                                    y[-1][-1][preTrials+1:preTrials+1+k] = post[:k]
                        y[-1] = np.nanmean(y[-1],axis=0)
                    m = np.nanmean(y,axis=0)
                    s = np.nanstd(y,axis=0)/(len(y)**0.5)
                    ax.plot(x,m,color=clr,label=context)
                    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=10)
                ax.set_xticks(np.arange(-20,60,20))
                ax.set_yticks([0,0.5,1])
                ax.set_xlim([-preTrials-0.5,postTrials+0.5])
                ax.set_ylim([0,1.01])
                if i==len(fixedParamNames)-1:
                    ax.set_xlabel('Trials after block switch')
                if j==0:
                    ax.set_ylabel(ylbl)
                if fixedParam=='Full model':
                    title = fixedParam+', '+blockLabel
                else:
                    title = fixedParam+'='+str(fixedVal)
                ax.set_title(title)
                if var=='pContext' and i==0 and j==1:
                    ax.legend(bbox_to_anchor=(1,1))
        plt.tight_layout()



# cluster fit comparison of model and mice
for modelType in modelTypes:
    for clustInd,clust in enumerate(clusterIds): 
        fig = plt.figure(figsize=(8,10))
        fig.suptitle(('alphaStim=0' if 'alphaStim' in fixedParam else 'full model') + ', cluster ' + str(clustInd+1))
        gs = matplotlib.gridspec.GridSpec(len(fixedParamNames[modelType])+1,2)
        stimNames = ('vis1','vis2','sound1','sound2')
        postTrials = 15
        x = np.arange(postTrials)+1
        for i,fixedParam in enumerate(('mice',)+fixedParamNames[modelType]):  
            d = modelData['clusters']
            for j,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks'))):
                ax = fig.add_subplot(gs[i,j])
                for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                    y = []
                    for mouse in d:
                        for session in d[mouse]:
                            if modelType in d[mouse][session]:
                                obj = sessionData[trainingPhase][mouse][session]
                                clustTrials = np.array(clustData['trialCluster'][mouse][session]) == clust
                                if np.any(clustTrials):
                                    if fixedParam == 'mice':
                                        resp = obj.trialResponse
                                    else:
                                        resp = d[mouse][session][modelType]['simulation'][fixedParamNames[modelType].index(fixedParam)][clustInd]
                                    if not np.all(np.isnan(resp)):
                                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                            if rewStim==rewardStim:
                                                trials = clustTrials & (obj.trialBlock==blockInd+1) & (obj.trialStim==stim) & ~obj.autoRewardScheduled 
                                                if np.any(trials):
                                                    y.append(np.full(postTrials,np.nan))
                                                    post = resp[trials]
                                                    k = min(postTrials,post.size)
                                                    y[-1][:k] = post[:k]
                    m = np.nanmean(y,axis=0)
                    s = np.nanstd(y,axis=0)/(len(y)**0.5)
                    ax.plot(x,m,color=clr,ls=ls,label=stim)
                    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks(np.arange(-5,20,5))
                ax.set_yticks([0,0.5,1])
                ax.set_xlim([0.5,postTrials+0.5])
                ax.set_ylim([0,1.01])
                if i==len(modelTypes):
                    ax.set_xlabel('Trials after block switch cues')
                if j==0:
                    ax.set_ylabel(('Response rate' if modelType=='mice' else 'Prediction'))
                if modelType=='mice':
                    title = 'mice, ' if rewardStim=='vis1' else ''
                    title += blockLabel+' (n='+str(len(y))+')'
                else:
                    title = modelType
                ax.set_title(title)
                if i==0 and j==1:
                    ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.tight_layout()
        

# cluster fit comparison of model and mice, combined block types and targets only
for modelType in modelTypes:
    for clustInd,clust in enumerate(clusterIds): 
        fig = plt.figure(figsize=(5,10))
        postTrials = 15
        x = np.arange(postTrials)+1
        for i,fixedParam in enumerate(('mice',)+fixedParamNames[modelType]):  
            d = modelData['clusters']
            ax = fig.add_subplot(len(fixedParamNames[modelType])+1,1,i+1)
            for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality')[:2],'gmgm',('-','-','--','--')):
                y = []
                for mouse in d:
                    for session in d[mouse]:
                        if modelType in d[mouse][session]:
                            obj = sessionData[trainingPhase][mouse][session]
                            clustTrials = np.array(clustData['trialCluster'][mouse][session]) == clust
                            if np.any(clustTrials):
                                if fixedParam == 'mice':
                                    resp = obj.trialResponse
                                else:
                                    resp = d[mouse][session][modelType]['simulation'][fixedParamNames[modelType].index(fixedParam)][clustInd]
                                if not np.all(np.isnan(resp)):
                                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                        stim = np.setdiff1d(obj.blockStimRewarded,rewStim)[0] if 'unrewarded' in stimLbl else rewStim
                                        trials = clustTrials & (obj.trialBlock==blockInd+1) & (obj.trialStim==stim) & ~obj.autoRewardScheduled 
                                        if np.any(trials):
                                            y.append(np.full(postTrials,np.nan))
                                            post = resp[trials]
                                            k = min(postTrials,post.size)
                                            y[-1][:k] = post[:k]
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x,m,color=clr,ls=ls,label=stim)
                ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks(np.arange(-5,20,5))
            ax.set_yticks([0,0.5,1])
            ax.set_xlim([0.5,postTrials+0.5])
            ax.set_ylim([0,1.01])
            if i==len(modelTypes):
                ax.set_xlabel('Trials after block switch cues')
            ax.set_ylabel(('Response rate' if modelType=='mice' else 'Prediction'))
            if fixedParam=='mice':
                title = 'mice' + ' (cluster' + str(clust) + ', n='+str(len(y))+' blocks)'
            else:
                title = modelType + ', ' + str(fixedParam)
            ax.set_title(title)
            if i==0:
                ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.tight_layout()

  
# cluster fit log loss
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for modelType,clr in zip(modelTypes,modelTypeColors):
    for j,clust in enumerate(clusterIds):
        val = []
        for mouse in modelData['clusters'].values():
            for session in mouse.values():
                if modelType in session:
                    val.extend(session[modelType]['logLossTest'][0][j])
        val = np.array(val)
        val[np.isinf(val)] = np.nan
        m = np.nanmean(val)
        s = np.nanstd(val)/(len(val)**0.5)
        ax.plot(j,m,'o',mec=clr,mfc='none',ms=10,mew=2,label=(modelType if j==0 else None))
        ax.plot([j,j],[m-s,m+s],color=clr,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(nClusters))
ax.set_xticklabels(np.arange(nClusters)+1)
ax.set_xlim([-0.25,nClusters-0.25])
# ax.set_ylim([0.35,0.6])
ax.set_xlabel('Cluster')
ax.set_ylabel('Negative log-likelihood')
ax.legend(loc='upper left',bbox_to_anchor=(1,1))
plt.tight_layout()
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([-1,nClusters+1],[0,0],'k--')
for modelType,clr in zip(modelTypes,modelTypeColors):
    for j in range(nClusters):
        val = []
        for mouse in modelData['clusters'].values():
            for session in mouse.values():
                val.extend([b-a for a,b in zip(session[modelType]['logLossTest'][0][j],session[modelType]['logLossTest'][1][j])])
        val = np.array(val)
        val[np.isinf(val)] = np.nan
        m = np.nanmean(val)
        s = np.nanstd(val)/(len(val)**0.5)
        ax.plot(j,m,'o',mec=clr,mfc='none',ms=10,mew=2,label=(modelType if j==0 else None))
        ax.plot([j,j],[m-s,m+s],color=clr,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(nClusters))
ax.set_xticklabels(np.arange(nClusters)+1)
ax.set_xlim([-0.25,nClusters-0.25])
ax.set_xlabel('Cluster')
ax.set_ylabel('$\Delta$ Negative log-likelihood')
ax.legend(loc='upper left',bbox_to_anchor=(1,1))
plt.tight_layout()


# cluster fit param values
for modelType in modelTypes:
    fig = plt.figure(figsize=(14,11))
    gs = matplotlib.gridspec.GridSpec(len(fixedParamNames[modelType]),len(paramNames[modelType]))
    for i,fixedParam in enumerate(fixedParamNames[modelType]):
        for j,(param,xlim) in enumerate(zip(paramNames[modelType],paramBounds[modelType])):
            ax = fig.add_subplot(gs[i,j])
            for clustInd,(clust,clr) in enumerate(zip(clusterIds,clusterColors)):
                paramVals = []
                for mouse in modelData['clusters'].values():
                    for session in mouse.values():
                        if modelType in session:
                            vals = session[modelType]['params'][i][clustInd]
                            if not np.all(np.isnan(vals)):
                                paramVals.append(vals[j])
                if len(np.unique(paramVals)) > 1:
                    dsort = np.sort(paramVals)
                    cumProb = np.array([np.sum(dsort<=s)/dsort.size for s in dsort])
                    ax.plot(dsort,cumProb,color=clr,label='cluster '+str(clust))
                    print(fixedParam,clust,param,np.median(paramVals))
                else:
                    ax.plot(paramVals[0],1,'o',mfc=clr,mec=clr)
                    print(fixedParam,clust,param,paramVals[0])
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([xlim[0]-0.02,xlim[1]+0.02])
            ax.set_ylim([0,1.01])
            if j>0:
                ax.set_yticklabels([])
            if i<len(modelTypes)-1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(param)
            if j==0 and i==len(modelTypes)//2:
                ax.set_ylabel('Cum. Prob.')
            if j==len(paramNames[modelTypes[-1]])//2:
                ax.set_title(modelType)
            if i==1 and j==len(paramNames[modelTypes[-1]])-1:
                ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    plt.tight_layout()
    

    
# decoder confidence correlation with model
decodeDataPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\CO decoding results\logreg_2024-11-27_re_concat_1\decoder_confidence_all_trials_all_units.pkl"
df = pd.read_pickle(decodeDataPath)

areas = ('FRP','ORBl','ORBm','ORBvl','PL','MOs','ACAd','ACAv','CP','STR','GPe','SNr','SCm','MRN')

area = 'MOs'

trainingPhase = 'ephys'
modelType = 'contextRL'
md = modelData[trainingPhase]
for mouse in md.keys():
    for session in md[mouse].keys():
        sessionName = mouse + '_' + datetime.datetime.strptime(session[:8],'%Y%m%d').strftime('%Y-%m-%d')
        decoderConfVis = df[(df['session']==sessionName) & (df['area']==area) & ((df['probe']=='') | (df['probe']=='all'))]['confidence']
        if len(decoderConfVis) > 0:
            decoderConfVis = np.array(decoderConfVis)[0]
            decoderConfVis = ((decoderConfVis / np.max(np.absolute(decoderConfVis))) + 1) / 2
            
            obj = sessionData[trainingPhase][mouse][session]
            s = md[mouse][session][modelType]
            pVis = s['pContext'][fixedParamNames[modelType].index('Full model')][:,0]
            qPerVis,qPerAud = s['qPerseveration'][fixedParamNames[modelType].index('Full model')][:,[0,2]].T
            perseveration = (qPerVis-qPerAud+1)/2
            
            fig = plt.figure(figsize=(12,4))
            ax = fig.add_subplot(1,1,1)
            x = np.arange(obj.nTrials) + 1
            ax.plot([0,x[-1]+1],[0.5,0.5],'--',color='0.5')
            blockStarts = np.where(obj.blockTrial==0)[0]
            for i,(b,rewStim) in enumerate(zip(blockStarts,obj.blockStimRewarded)):
                if rewStim == 'vis1':
                    w = blockStarts[i+1] - b if i < 5 else obj.nTrials - b
                    ax.add_patch(matplotlib.patches.Rectangle([b+1,0],width=w,height=1,facecolor='0.5',edgecolor=None,alpha=0.1,zorder=0))
            ax.plot(x,pVis,'k',label='prob vis')
            # ax.plot(x,qPerVis,'r',label='perseveration vis')
            # ax.plot(x,qPerAud,'b',label='perseveration aud')
            ax.plot(x,perseveration,'g',label='perseveration')
            ax.plot(x,decoderConfVis,'c',label='decoder conf vis')
            y = 1.05
            for stim,clr in zip(('vis1','sound1'),'rb'):
                for resp in (True,False):
                    trials = np.where((obj.trialStim==stim) & (obj.trialResponse if resp else ~obj.trialResponse))[0] + 1
                    ax.vlines(trials,y-0.02,y+0.02,color=clr,alpha=(1 if resp else 0.5))
                    y += 0.05
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=12)
            ax.set_xlim([0,x[-1]+1])
            ax.set_yticks([0,0.5,1])
            # ax.set_ylim([0,1.25])
            ax.set_xlabel('Trial',fontsize=12)
            ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=12)
            plt.tight_layout()
            assert(False)
    



areas = ('FRP','ORBl','ORBm','ORBvl','PL','ILA','ACAd','ACAv','MOs','MOp','AId','AIp','AIv','RSPd','RSPv','VISp','AUDp',
         'CP','STR','ACB','GPe','SNr','SCs','SCm','MRN','SNc','VTA')

trainingPhase = 'ephys'
modelType = 'contextRL'
md = modelData[trainingPhase]
labels = ('pContext','perseveration','decoder conf')
corr = {area: [] for area in areas}
for area in areas:
    for mouse in md.keys():
        for session in md[mouse].keys():
            sessionName = mouse + '_' + datetime.datetime.strptime(session[:8],'%Y%m%d').strftime('%Y-%m-%d')
            decoderConfVis = df[(df['session']==sessionName) & (df['area']==area) & ((df['probe']=='') | (df['probe']=='all'))]['confidence']
            if len(decoderConfVis) > 0:
                decoderConf = np.array(decoderConfVis)[0]
                obj = sessionData[trainingPhase][mouse][session]
                s = md[mouse][session][modelType]
                pVis = s['pContext'][fixedParamNames[modelType].index('Full model')][:,0] - 0.5
                qPerVis,qPerAud = s['qPerseveration'][fixedParamNames[modelType].index('Full model')][:,[0,2]].T
                perseveration = qPerVis - qPerAud
                d = (pVis,perseveration,decoderConf)
                c = np.zeros((len(d),len(d)))
                for i,a in enumerate(d):
                    for j,b in enumerate(d):
                        c[i,j] = np.corrcoef(a,b)[0,1]
                corr[area].append(c)
    corr[area] = np.array(corr[area])

for area in areas:    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)  
    c = np.mean(corr[area],axis=0)     
    cmax = np.nanmax(np.absolute(c))
    im = ax.imshow(c,cmap='bwr',clim=(-cmax,cmax))
    cb = plt.colorbar(im,ax=ax,fraction=0.01,pad=0.04)
    # cb.set_ticks()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(area)
    plt.tight_layout()

for area in areas:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = (0,1)
    ax.plot(alim,alim,'--',color='0.5')
    ax.plot(corr[area][:,2,0],corr[area][:,2,1],'o',mec='k',mfc='none')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    # ax.set_xlabel('run speed, visual rewarded blocks (cm/s)')
    # ax.set_ylabel('run speed, auditory rewarded blocks (cm/s)')
    ax.set_title(area)
    plt.tight_layout()
    
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
for x,area in enumerate(areas):
    for i,offset in zip((0,1),(-0.05,0.05)):
       c =  corr[area][:,2,i]
       m = c.mean()
       s = c.std() / (len(c)**0.5)
       ax.plot(x+offset,m,'o',mec='k',mfc=('k' if i==0 else 'none'))
       ax.plot([x+offset]*2,[m-s,m+s],'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(len(areas)))
ax.set_xticklabels(areas)
ax.set_ylim([0,1])
plt.tight_layout()

















