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

clustData = np.load(os.path.join(baseDir,'clustData.npy'),allow_pickle=True).item()


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
if fitClusters:
    clusterIds = (3,4,5,6)
    nClusters = len(clusterIds)
    clusterColors = ([clr for clr in 'rgkbmcy']+['0.6'])[:nClusters]
    trainingPhases = ('clusters',)
    trainingPhaseColors = 'k'
else:
    trainingPhases = ('after learning',)
    # trainingPhases = ('nogo','noAR','rewardOnly','no reward') 
    # trainingPhases = ('opto',)
    trainingPhaseColors = 'mgrbck'
if 'opto' in trainingPhases:
    modelTypes = ('contextRL',)
else:
    modelTypes = ('contextRL',)
modelTypeColors = 'rb'

modelParams = {'visConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
               'audConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
               'biasAction': {'bounds':(-1,1), 'fixedVal': 0},
               'alphaContext': {'bounds':(0,1), 'fixedVal': np.nan},
               'alphaContextNeg': {'bounds': (0,1), 'fixedVal': np.nan},
               'tauContext': {'bounds': (1,240), 'fixedVal': np.nan},
               'blockTiming': {'bounds': (0,1), 'fixedVal': np.nan},
               'blockTimingShape': {'bounds': (0.5,4), 'fixedVal': np.nan},
               'wReinforcement': {'bounds': (0,30), 'fixedVal': 0},
               'alphaReinforcement': {'bounds': (0,1), 'fixedVal': np.nan},
               'alphaReinforcementNeg': {'bounds': (0,1), 'fixedVal': np.nan},
               'tauReinforcement': {'bounds': (1,10000), 'fixedVal': np.nan},
               'wPerseveration': {'bounds': (0,30), 'fixedVal': 0},
               'alphaPerseveration': {'bounds': (0,1), 'fixedVal': np.nan},
               'tauPerseveration': {'bounds': (1,10000), 'fixedVal': np.nan},
               'alphaReward': {'bounds': (0,1), 'fixedVal': np.nan},
               'tauReward': {'bounds': (1,60), 'fixedVal': np.nan}}

paramNames = {}
fixedParamNames = {}
fixedParamLabels = {}
for modelType in modelTypes:
    if modelType == 'basicRL':
        paramNames[modelType] = ('visConfidence','audConfidence','biasAction','wReinforcement','alphaReinforcement',
                                 'wPerseveration','alphaPerseveration','alphaReward','tauReward')
    else:
        paramNames[modelType] = ('visConfidence','audConfidence','biasAction','alphaContext','tauContext','blockTiming','blockTimingShape',
                                 'wReinforcement','alphaReinforcement','tauReinforcement','wPerseveration','alphaPerseveration','tauPerseveration','alphaReward','tauReward')
    fixedParamNames[modelType] = ('Full model',)
    fixedParamLabels[modelType] = ('Full model',)
    if fitClusters:
        if modelType == 'basicRL':
            fixedParamNames[modelType] += ('alphaReinforcement',)
        elif modelType == 'contextRL':
            fixedParamNames[modelType] += ('decayContext','blockTiming',('decayContext','blockTiming'))
    elif 'opto' in trainingPhases:
        fixedParamNames[modelType] += ('betaActionOpto','biasActionOpto')
    elif modelType in ('nogo','noAR'):
        pass
    else:
        if modelType == 'basicRL':
            fixedParamNames[modelType] += ('alphaReinforcement','alphaPerseveration','alphaReward')
            fixedParamLabels[modelType] += ('no state-action\nvalue learning','no perseveration','no reward\nbias')
        else:
            pass
            # fixedParamNames[modelType] += ('tauContext','blockTiming',('tauContext','blockTiming'),'alphaReinforcement','alphaPerseveration','alphaReward')
            # fixedParamLabels[modelType] += ('no context\nforgetting','no block\ntiming','no context\nforgetting or\nblock timing',
            #                                 'no state-action\nvalue learning','no perseveration','no reward\nbias')


modelTypeParams = {}
modelData = {phase: {} for phase in trainingPhases}
dirPath = os.path.join(baseDir,'RLmodel')
if trainingPhases[0] == 'opto':
    dirPath = os.path.join(dirPath,'opto')
elif fitClusters:
    dirPath = os.path.join(dirPath,'clusters')
filePaths = glob.glob(os.path.join(dirPath,'*.npz'))
for fileInd,f in enumerate(filePaths):
    print(fileInd)
    mouseId,sessionDate,sessionTime,trainingPhase,modelType = os.path.splitext(os.path.basename(f))[0].split('_')
    if trainingPhase not in trainingPhases or modelType not in modelTypes:
        continue
    session = sessionDate+'_'+sessionTime
    with np.load(f,allow_pickle=True) as data:
        if 'params' not in data:
            continue
        params = data['params']
        logLoss = data['logLoss']
        termMessage = data['terminationMessage']
        if 'trainSessions' in data:
            trainSessions = data['trainSessions']
        else:
            trainSessions = None
        if modelType not in modelTypeParams:
            modelTypeParams[modelType] = {key: val for key,val in data.items() if key not in ('params','logLoss','terminationMessage','trainSessions')}
            if 'optoLabel' in modelTypeParams[modelType] and len(modelTypeParams[modelType]['optoLabel'].shape)==0:
                modelTypeParams[modelType]['optoLabel'] = None
    d = modelData[trainingPhase]
    p = {'params': params, 'logLossTrain': logLoss, 'terminationMessage': termMessage, 'trainSessions': trainSessions}
    if mouseId not in d:
        d[mouseId] = {session: {modelType: p}}
    elif session not in d[mouseId]:
        d[mouseId][session] = {modelType: p}
    elif modelType not in d[mouseId][session]:
        d[mouseId][session][modelType] = p


# print fit termination message
# for trainingPhase in trainingPhases:
#     for mouse in modelData[trainingPhase]:
#         for session in modelData[trainingPhase][mouse]:
#             for modelType in modelTypes:
#                 print(modelData[trainingPhase][mouse][session][modelType]['terminationMessage'])


## get experiment data and model variables
sessionData = {phase: {} for phase in trainingPhases}
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
            d[mouse][session]['Naive'] = {'logLossTest': sklearn.metrics.log_loss(obj.trialResponse,naivePrediction)}
            for modelType in modelTypes:
                if modelType not in d[mouse][session]:
                    continue
                s = d[mouse][session][modelType]
                if fitClusters:
                    s['pContext'] = [[] for _ in range(len(fixedParamNames[modelType]))]
                    s['qReinforcement'] = copy.deepcopy(s['pContext'])
                    s['qReward'] = copy.deepcopy(s['pContext'])
                    s['qTotal'] = copy.deepcopy(s['pContext'])
                    s['prediction'] = copy.deepcopy(s['pContext'])
                    s['logLossTest'] = [[[] for _ in range(nClusters)] for _ in range(len(fixedParamNames[modelType]))]
                    s['simulation'] = copy.deepcopy(s['pContext'])
                    s['simAction'] = copy.deepcopy(s['pContext'])
                    s['logLossSimulation'] = []
                    for i,prms in enumerate(s['params']):
                        for clustInd,params in enumerate(prms):
                            if np.all(np.isnan(params)):
                                pContext,qReinforcement,qPerseveration,qReward,qTotal,pAction,action,pSimulate = [np.nan] * 7
                                simAction = []
                            else:
                                pContext,qReinforcement,qPerseveration,qReward,qTotal,pAction,action = [val[0] for val in runModel(obj,*params,**modelTypeParams[modelType])]
                                pSimulate,simAction = runModel(obj,*params,useChoiceHistory=False,nReps=10,**modelTypeParams[modelType])[-2:]
                                pSimulate = np.mean(pSimulate,axis=0)
                            s['pContext'][i].append(pContext)
                            s['qReinforcement'][i].append(qReinforcement)
                            s['qReward'][i].append(qReward)
                            s['qTotal'][i].append(qTotal)
                            s['prediction'][i].append(pAction)
                            s['simulation'][i].append(pSimulate)
                            s['simAction'][i].append(simAction)
                            resp = obj.trialResponse
                            pred = pAction
                            if not np.any(np.isnan(pred)):
                                clustTrials = clustData['trialCluster'][mouse][session] == clustInd+1
                                for blockInd in range(6):
                                    trials = clustTrials & (obj.trialBlock==blockInd+1)
                                    if np.any(trials):
                                        if np.all(resp[trials]):
                                            s['logLossTest'][i][clustInd].append(np.mean(np.log(pred[trials])))
                                        elif not np.any(resp[trials]):
                                            s['logLossTest'][i][clustInd].append(np.mean(np.log(1-pred[trials])))
                                        else:
                                            s['logLossTest'][i][clustInd].append(sklearn.metrics.log_loss(resp[trials],pred[trials]))
                else:
                    s['pContext'] = []
                    s['qReinforcement'] = []
                    s['qPerseveration'] = []
                    s['qReward'] = []
                    s['qTotal'] = []
                    s['prediction'] = []
                    s['logLossTest'] = []
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
                        s['logLossTest'].append(sklearn.metrics.log_loss(obj.trialResponse[trials],pAction[trials]))
                        pContext,qReinforcement,qPerseveration,qReward,qTotal,pAction,action = runModel(obj,*params,useChoiceHistory=False,nReps=10,**modelTypeParams[modelType])
                        s['simulation'].append(np.mean(pAction,axis=0))
                        s['simAction'].append(action)
                        s['simPcontext'].append(pContext)
                        s['simQreinforcement'].append(qReinforcement)
                        s['simQperseveration'].append(qPerseveration)
                        s['logLossSimulation'].append(np.mean([sklearn.metrics.log_loss(obj.trialResponse,p) for p in pAction]))


## simulate loss-of-function
for trainingPhase in trainingPhases:
    d = modelData[trainingPhase]
    for mouse in d:
        for session in d[mouse]:
            for modelType in modelTypes:
                obj = sessionData[trainingPhase][mouse][session]
                s = d[mouse][session][modelType]
                s['simLossParam'] = []
                s['simLossParamAction'] = []    
                s['simLossParamPcontext'] = []   
                for fixedParam in fixedParamNames[modelType]:
                    params = s['params'][fixedParamNames[modelType].index('Full model')].copy()
                    if fixedParam != 'Full model':
                        for prm in (fixedParam if isinstance(fixedParam,tuple) else (fixedParam,)):
                            params[paramNames[modelType].index(prm)] = fixedParamValues[modelType][fixedParamNames[modelType].index(fixedParam)]
                    pContext,qReinforcement,qReward,qTotal,pAction,action = runModel(obj,*params,useChoiceHistory=False,nReps=1,**modelTypeParams[modelType])
                    s['simLossParam'].append(pAction[0])
                    s['simLossParamAction'].append(action[0])
                    s['simLossParamPcontext'].append(pContext[0])


## compare model prediction and model simulation  
modelInd = 0       
for trainingPhase in trainingPhases:
    d = modelData[trainingPhase]
    for modelType in modelTypes:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        pred = []
        sim = []
        for mouse in d:
            for session in d[mouse]:
                if modelType in d[mouse][session]:
                    s = d[mouse][session][modelType]
                    pred.append(np.exp(-s['logLossTest'][modelInd]))
                    sim.append(np.exp(-s['logLossSimulation'][modelInd]))
        ax.plot(pred,sim,'o',mec='k',mfc=None,alpha=0.25)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_aspect('equal')
        ax.set_xlabel('likelihood of model prediction')
        ax.set_ylabel('likelihood of model simulation')
        slope,yint,rval,pval,stderr = scipy.stats.linregress(pred,sim)
        ax.set_title(trainingPhase+', '+modelType+'\nr^2 = '+str(round(rval**2,2)))
 
for trainingPhase in trainingPhases:
    d = modelData[trainingPhase]
    for modelType in modelTypes:
        r = []  
        for mouse in d:
            for session in d[mouse]:
                if modelType in d[mouse][session]:
                    s = d[mouse][session][modelType]
                    pred = s['prediction'][0]
                    sim = s['simulation'][0]
                    slope,yint,rval,pval,stderr = scipy.stats.linregress(pred,sim)
                    r.append(rval**2)
        print(trainingPhase,modelType,round(np.median(r),2))
        

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
                    obj = sessionData[trainingPhase][mouse][session]
                    if fixedParam == 'mice':
                        resp = obj.trialResponse
                    else:
                        if modelType not in d[mouse][session]:
                            continue
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


for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(len(fixedParamLabels[modelType])+1)
    for trainingPhase,clr in zip(trainingPhases,'mg'):
        d = performanceData[trainingPhase][modelType]
        d = [d[lbl]['dprime'] for lbl in d]
        mean = np.mean(d,axis=1)
        sem = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(x,mean,'o',mec=clr,mfc=None,label=trainingPhase)
        for xi,m,s in zip(x,mean,sem):
            ax.plot([xi,xi],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(('Mice',)+fixedParamLabels[modelType])
    ax.set_xlim([-0.25,len(x)+0.25])
    ax.set_ylim([0,2.2])
    ax.set_ylabel('cross-modal d\'')
    ax.legend()
    plt.tight_layout()
    
for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(len(fixedParamLabels[modelType])+1)
    ax.plot([-1,len(x)+1],[0,0],'k--')
    for trainingPhase,clr in zip(trainingPhases,'mg'):
        d = performanceData[trainingPhase][modelType]
        respFirst = [d[lbl]['respFirst'] for lbl in d]
        respLast = [d[lbl]['respLast'] for lbl in d]
        d = np.array(respFirst) - np.array(respLast)
        mean = np.mean(d,axis=1)
        sem = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(x,mean,'o',mec=clr,mfc=None,label=trainingPhase)
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
for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    xticks = np.arange(len(fixedParamLabels[modelType])+1)
    xlim = [-0.25,xticks[-1]+0.25]
    for trainingPhase,clr in zip(trainingPhases,'mg'):
        d = modelData[trainingPhase]
        naive = np.array([np.mean([np.exp(-np.array(session['Naive']['logLossTest'])) for session in mouse.values()],axis=0) for mouse in d.values()])
        lh = np.array([np.mean([np.exp(-np.array(session[modelType]['logLossTest'])) for session in mouse.values() if modelType in session],axis=0) for mouse in d.values()])
        lh = np.concatenate((naive[:,None],lh),axis=1)
        mean = np.mean(lh,axis=0)
        sem = np.std(lh,axis=0)/(len(lh)**0.5)
        x = np.arange(len(mean))
        ax.plot(x,mean,'o',mec=clr,mfc=None,alpha=0.5,label=trainingPhase)
        for xi,m,s in zip(x,mean,sem):
            ax.plot([xi,xi],[m-s,m+s],color=clr,alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(('Naive\n(fixed response\nprobability)',)+fixedParamLabels[modelType])
    ax.set_xlim(xlim)
    ax.set_ylim([0.5,0.8])
    ax.set_ylabel('Likelihood')
    ax.legend()
    plt.tight_layout()
                
                
## plot param values
for modelType in modelTypes:
    fig = plt.figure(figsize=(11,5))
    gs = matplotlib.gridspec.GridSpec(len(fixedParamNames[modelType]),len(paramNames[modelType]))
    for i,fixedParam in enumerate(fixedParamNames[modelType]):
        for j,param in enumerate(paramNames[modelType]):
            ax = fig.add_subplot(gs[i,j])
            for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
                d = modelData[trainingPhase]
                if len(d) > 0:
                    prmInd = list(modelParams.keys()).index(param)
                    paramVals = np.array([np.mean([session[modelType]['params'][i,prmInd] for session in mouse.values() if modelType in session]) for mouse in d.values()])
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
            ax.tick_params(direction='out',top=False,right=False)
            if param in ('tauReinforcement','tauPerseveration'):
                ax.set_xscale('log')
            xlim = modelParams[param]['bounds']
            ax.set_xlim([xlim[0]-0.02,xlim[1]+0.02])
            ax.set_ylim([0,1.01])
            if j>0:
                ax.set_yticklabels([])
            if i<len(fixedParamNames[modelType])-1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(param)
            if j==0 and i==len(fixedParamNames[modelType])//2:
                ax.set_ylabel('Cum. Prob.')
            if j==len(paramNames[modelType])//2:
                ax.set_title(str(fixedParam))
            # if i==0 and j==len(paramNames[modelType])-1:
            #     ax.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()


## compare model and mice
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
            
# less plots
var = 'simulation'
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 5
postTrials = 15
x = np.arange(-preTrials,postTrials+1)
for modelType in ('basicRL',): #modelTypes:
    for trainingPhase in trainingPhases:
        for fixedParam in ('mice','Full model','alphaReinforcement','rewardBias'):
            if fixedParam == 'mice' and modelType=='basicRL':
                d = sessionData[trainingPhase]
            elif fixedParam in fixedParamNames[modelType]:
                d = modelData[trainingPhase]
            else:
                continue
            fig = plt.figure(figsize=(8,4.5))
            if fixedParam=='mice':
                title = 'Mice, '+trainingPhase
            else:
                title = modelType + ', ' + trainingPhase + ', ' + fixedParam
            fig.suptitle(title)
            for i,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks'))):
                ax = fig.add_subplot(1,2,i+1)
                for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
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
                                if rewStim==rewardStim and blockInd > 0:
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
                    ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
                    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks(np.arange(-5,20,5))
                ax.set_yticks([0,0.5,1])
                ax.set_xlim([-preTrials-0.5,postTrials+0.5])
                ax.set_ylim([0,1.01])
                ax.set_xlabel('Trials after block switch')
                if i==0:
                    ax.set_ylabel('Response rate')
                ax.set_title(blockLabel)
                if i==1:
                    ax.legend(loc='upper left',bbox_to_anchor=(1,1))
            plt.tight_layout()
            
# combine block types
var = 'simLossParam'
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for modelType in ('contextRLForgetting',): #modelTypes:
    for trainingPhase in ('after learning',): #trainingPhases:
        for fixedParam in ('mice','Full model','decayContext','rewardBias',('decayContext','rewardBias')):
            if fixedParam == 'mice' and modelType=='basicRL':
                d = sessionData[trainingPhase]
            elif fixedParam in fixedParamNames[modelType]:
                d = modelData[trainingPhase]
            else:
                continue
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(1,1,1)
            ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
            for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
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
                            if blockInd > 0:
                                stim = np.setdiff1d(obj.blockStimRewarded,rewStim)[0] if 'unrewarded' in stimLbl else rewStim
                                if 'non-target' in stimLbl:
                                    stim = stim[:-1]+'2'
                                trials = (obj.trialStim==stim) #& ~obj.autoRewardScheduled
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
            ax.tick_params(direction='out',top=False,right=False,labelsize=18)
            ax.set_xticks([-5,-1,5,9,14,19])
            ax.set_xticklabels([-5,-1,1,5,10,15])
            ax.set_yticks([0,0.5,1])
            ax.set_xlim([-preTrials-0.5,postTrials-0.5])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Trials after block switch',fontsize=20)
            ax.set_ylabel('Response rate',fontsize=20)
            if fixedParam=='mice':
                title = 'Mice, '+trainingPhase
            else:
                title = modelType + ', ' + trainingPhase + ', ' + str(fixedParam)
            # ax.set_title(title)
            ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
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
blockEpochs = ('full',) #'first half','last half')
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
                        obj = sessionData[trainingPhase][mouse][session]
                        if modelType=='mice': 
                            trialResponse = [obj.trialResponse]
                        else:
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
for modelType in modTypes:
    for fixedParam in fxdPrms[modelType]:
        for phase in trainingPhases:
            for epoch in ('full',):
                for prevTrialType in prevTrialTypes:    
                    fig = plt.figure()#(figsize=(12,6))
                    ax = fig.add_subplot(1,1,1)
                    clrs = 'gmgm'
                    for stim,clr,ls in zip(stimType,clrs,('-','-','--','--')):
                        n = []
                        p = []
                        for d,r in zip(timeSince[modelType][fixedParam][phase][epoch][prevTrialType][stim],respNorm[modelType][fixedParam][phase][epoch][stim]):
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
                    ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=14)
                    ax.set_ylabel('Response rate\n(difference from within-block mean)',fontsize=14)
                    # ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=16)
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
autoCorr = {modelType: {fixedParam: {phase: {epoch: [[[] for _  in range(len(modelData[phase]))] for _ in range(4)] for epoch in blockEpochs} for phase in trainingPhases} for fixedParam in fxdPrms[modelType]} for modelType in modTypes}
autoCorrDetrend = copy.deepcopy(autoCorr)
corrWithin = {modelType: {fixedParam: {phase: {epoch: [[[[] for _  in range(len(modelData[phase]))] for _ in range(4)] for _ in range(4)] for epoch in blockEpochs} for phase in trainingPhases} for fixedParam in fxdPrms[modelType]} for modelType in modTypes}
corrWithinDetrend = copy.deepcopy(corrWithin)
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
                            for epoch in blockEpochs:
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
                                        autoCorr[modelType][fixedParam][phase][epoch][i][m].append(corr)
                                        corrDetrend,corrRawDetrend = getCorrelation(r,r,rs,rs,100,detrendOrder=2)
                                        autoCorrDetrend[modelType][fixedParam][phase][epoch][i][m].append(corrDetrend)
                                    
                                    r = resp[:,blockTrials]
                                    rs = respShuffled[:,blockTrials]
                                    for i,(r1,rs1) in enumerate(zip(r,rs)):
                                        for j,(r2,rs2) in enumerate(zip(r,rs)):
                                            if np.count_nonzero(r1) >= minTrials and np.count_nonzero(r2) >= minTrials:
                                                corr,corrRaw = getCorrelation(r1,r2,rs1,rs2)
                                                corrWithin[modelType][fixedParam][phase][epoch][i][j][m].append(corr)
                                                corrDetrend,corrRawDetrend = getCorrelation(r1,r2,rs1,rs2,detrendOrder=2)
                                                corrWithinDetrend[modelType][fixedParam][phase][epoch][i][j][m].append(corrDetrend)
                                
                for i in range(4):
                    for m in range(len(modelData[phase])):
                        autoCorrMat[modelType][fixedParam][phase][epoch][i,m] = np.nanmean(autoCorr[modelType][fixedParam][phase][epoch][i][m],axis=0)
                        autoCorrDetrendMat[modelType][fixedParam][phase][epoch][i,m] = np.nanmean(autoCorrDetrend[modelType][fixedParam][phase][epoch][i][m],axis=0)
                        
                for i in range(4):
                    for j in range(4):
                        for m in range(len(modelData[phase])):
                            corrWithinMat[modelType][fixedParam][phase][epoch][i,j,m] = np.nanmean(corrWithin[modelType][fixedParam][phase][epoch][i][j][m],axis=0)
                            corrWithinDetrendMat[modelType][fixedParam][phase][epoch][i,j,m] = np.nanmean(corrWithinDetrend[modelType][fixedParam][phase][epoch][i][j][m],axis=0)

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')

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

phase = 'after learning'
for modelType in ('mice','contextRL'):
    for fixedParam in fxdPrms[modelType]:
        fig = plt.figure(figsize=(12,10))          
        gs = matplotlib.gridspec.GridSpec(4,4)
        x = np.arange(1,200)
        for i,ylbl in enumerate(stimLabels):
            for j,xlbl in enumerate(stimLabels[:4]):
                ax = fig.add_subplot(gs[i,j])
                mat = corrWithinDetrendMat[modelType][fixedParam][phase]['full'][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr)
                ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=9)
                ax.set_xlim([0,20])
                ax.set_ylim([-0.02,0.04])
                if i==3:
                    ax.set_xlabel('Lag (trials)',fontsize=11)
                if j==0:
                    ax.set_ylabel(ylbl,fontsize=11)
                if i==0:
                    ax.set_title(xlbl,fontsize=11)
                # if i==0 and j==3:
                #     ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=11)
        plt.tight_layout()

                    

# no reward blocks, target stimuli only
for modelType in modelTypes:
    fig = plt.figure(figsize=(8,10))
    gs = matplotlib.gridspec.GridSpec(len(fixedParamNames[modelType])+1,2)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)  
    for i,(fixedParam,fixedVal) in enumerate(zip(('mice',) + fixedParamNames[modelType],(None,)+fixedParamValues[modelType])):
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

















