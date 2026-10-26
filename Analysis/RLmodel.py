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
from DynamicRoutingAnalysisUtils import DynRoutData, getSessionData, calcDprime, getBlockTrials, getResponseCorrelations
from RLmodelHPC import runModel


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Sam"


## find HPC out files with errors
outDir = '\\\\allen\\ai\\homedirs\\samg\\job_records'
outErrors = []
for f in glob.glob(os.path.join(outDir,'*.out')):
    with open(f,'r') as r:
        if len(r.readlines()) > 6:
            outErrors.append(f)


## get context beief from Ethan's decoder
f = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\CO decoding results\baseline_context_module_50_units_2026-08-20\predict_proba_per_session_structure_with_unit_ids.parquet"
df = pd.read_parquet(f)

contextBelief = {}
for session in df.session_id.unique():
    d = df[(df.session_id == session) & (df.structure=='context_module') & (df.balanced_accuracy_test > 0.7)]
    if len(d) > 0:
        fileName = 'DynamicRouting1_' + session.replace('-','') + '*.hdf5'
        filePath = glob.glob(os.path.join(r'\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data',session[:6],fileName))
        if len(filePath) > 0:
            obj = DynRoutData()
            obj.loadBehavData(filePath[0],lightLoad=True)
            if np.sum(np.array(obj.dprimeOtherModalGo) > 1) > 3:
                if obj.subjectName not in contextBelief:
                    contextBelief[obj.subjectName] = {}
                contextBelief[obj.subjectName][obj.startTime] = np.mean(d.predict_proba,axis=0)
                
# a = [contextBelief[m][s] for m in contextBelief for s in contextBelief[m]]

np.save(os.path.join(baseDir,'contextBelief.npy'),contextBelief)

# d = np.load(os.path.join(baseDir,'contextBelief.npy'),allow_pickle=True).item()


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
        if bi < 0.1:
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


## get fit params from HPC output
dirName = 'noiseSim'
if dirName == 'learning':
    trainingPhases = ('initial training','early learning','late learning','after learning')
    trainingPhaseColors = 'rmbgck'
    modelTypes = ('BasicRL','ContextRL')
    filesPerSession = 1
elif dirName == 'sessionCluster5':
    trainingPhases = ('odd','even')
    trainingPhaseColors = 'mg'
    modelTypes = ('BasicRL','ContextRL')
    filesPerSession = 1
elif dirName in ('agents','perseveration','noiseSim','contextBelief'):
    trainingPhases = ('after learning',)
    trainingPhaseColors = 'k'
    modelTypes = ('ContextRL',)
    filesPerSession = 1

modelTypeColors = 'rb'

modelParams = {'visConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
               'audConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
               'qInitVis': {'bounds': (0,1), 'fixedVal': 0},
               'qInitAud': {'bounds': (0,1), 'fixedVal': 0},
               'wContext': {'bounds': (0,30), 'fixedVal': 0},
               'alphaContext': {'bounds':(0,1), 'fixedVal': np.nan},
               'alphaContextNeg': {'bounds': (0,1), 'fixedVal': np.nan},
               'tauContext': {'bounds': (1,360), 'fixedVal': np.nan},
               'alphaContextReinforcement': {'bounds': (0,1), 'fixedVal': np.nan},
               'wReinforcement': {'bounds': (0,30), 'fixedVal': 0},
               'alphaReinforcement': {'bounds': (0,1), 'fixedVal': np.nan},
               'alphaReinforcementNeg': {'bounds': (0,1), 'fixedVal': np.nan},
               'tauReinforcement': {'bounds': (1,360), 'fixedVal': np.nan},
               'wPerseveration': {'bounds': (0,30), 'fixedVal': 0},
               'alphaPerseveration': {'bounds': (0,1), 'fixedVal': np.nan},
               'tauPerseveration': {'bounds': (1,360), 'fixedVal': np.nan},
               'wResponse': {'bounds': (0,30), 'fixedVal': 0},
               'alphaResponse': {'bounds': (0,1), 'fixedVal': np.nan},
               'tauResponse': {'bounds': (1,360), 'fixedVal': np.nan},
               'wReward': {'bounds': (0,30), 'fixedVal': 0},
               'alphaReward': {'bounds': (0,1), 'fixedVal': np.nan},
               'tauReward': {'bounds': (1,60), 'fixedVal': np.nan},
               'wBias': {'bounds': (0,30), 'fixedVal': 0},
               'muContextNoise': {'bounds': (-0.1,0.1), 'fixedVal': 0},
               'sigmaContextNoise': {'bounds': (0,0.1), 'fixedVal': 0}}
        
modelParamNames = list(modelParams.keys())
nModelParams = len(modelParamNames)

paramNames = {}
fixedParamNames = {}
lossParamNames = {}
for modelType in modelTypes:
    if modelType == 'BasicRL':
        coreFixedPrms = ['qInitVis','qInitAud','wContext','alphaContext','alphaContextNeg','tauContext','alphaContextReinforcement','alphaReinforcementNeg','tauReinforcement','wPerseveration','alphaPerseveration','tauPerseveration','wResponse','alphaResponse','tauResponse','muContextNoise','sigmaContextNoise']
    elif modelType == 'ContextRL':
        coreFixedPrms = ['qInitVis','qInitAud','alphaContextNeg','alphaContextReinforcement','wReinforcement','alphaReinforcement','alphaReinforcementNeg','tauReinforcement','wPerseveration','alphaPerseveration','tauPerseveration','wResponse','alphaResponse','tauResponse','muContextNoise','sigmaContextNoise']
    nPrms = nModelParams - len(coreFixedPrms)
    paramNames[modelType] = modelParamNames
    fixedParamNames[modelType] = ('Standard model',)
    lossParamNames[modelType] = ('Standard model',)
    if modelType == 'BasicRL':
        if dirName == 'learning':
            fixedParamNames[modelType] += ('-stimulus confidence','-reward agent','learning rate=0','asymmetric learning rates','state-action value forgetting','+context agent')
    elif modelType == 'ContextRL':
        if dirName == 'learning':
            fixedParamNames[modelType] += ('-stimulus confidence','-reward agent','-context forgetting','+context state-action value updates','+reinforcement agent','-context forgetting, +reinforcement agent')
        elif dirName == 'agents':
            fixedParamNames[modelType] += ('+perseveration','+perseveration, -context, -reward','+perseveration, -context','+perseveration, -neg alpha','+perseveration, -forgetting','+perseveration, -neg alpha, - forgetting')
        elif dirName == 'perseveration':
            fixedParamNames[modelType] += ('+stimulus-independent perseveration','+stimulus-dependent perseveration','-forgetting','-forgetting, +sd perseveration')
        elif dirName == 'noiseSim':
            fixedParamNames[modelType] += ('-context forgetting','-context forgetting, +context noise')
        elif dirName == 'contextBelief':
            fixedParamNames[modelType] += ('-context forgetting','+context belief')
            
            
modelData = {phase: {} for phase in trainingPhases}
dirPath = os.path.join(baseDir,'RLmodel',dirName)
filePaths = glob.glob(os.path.join(dirPath,'*.npz'))
modelOutputNames = ('params','logLossTrain','logLossTest','bicTrain','bicTest')
for fileInd,f in enumerate(filePaths):
    print(fileInd)
    fileParts = os.path.splitext(os.path.basename(f))[0].split('_')
    mouseId,sessionDate,sessionTime,trainingPhase = fileParts[:4]
    if filesPerSession > 1:
        modelType = '_'.join(fileParts[4:-1])
        fixedParamsIndex = int(fileParts[-1])
    else:
        modelType = '_'.join(fileParts[4:])
        fixedParamsIndex = None
    if trainingPhase not in trainingPhases or modelType not in modelTypes:
        continue
    with np.load(f,allow_pickle=True) as data:
        if 'params' not in data:
            continue
        params,logLossTrain,logLossTest,bicTrain,bicTest = [[np.median(d,axis=0) for d in data[key]] for key in modelOutputNames]
        paramsDict = {key: val for key,val in data.items() if key not in modelOutputNames}
    d = modelData[trainingPhase]
    mice = [mouseId]
    sessions = [[sessionDate+'_'+sessionTime]]
    for mouseId,mouseSessions in zip(mice,sessions):
        for session in mouseSessions:
            if mouseId not in d:
                d[mouseId] = {}
            if session not in d[mouseId]:
                d[mouseId][session] = {}
            if modelType not in d[mouseId][session]:
                if filesPerSession > 1:
                    d[mouseId][session][modelType] = {key: [None for _ in range(filesPerSession)] for key in modelOutputNames}
                    d[mouseId][session][modelType]['paramsDict'] = paramsDict
                else:
                    d[mouseId][session][modelType] = {'params': params, 'logLossTrain': logLossTrain, 'logLossTest': logLossTest, 'bicTrain': bicTrain, 'bicTest': bicTest, 'paramsDict': paramsDict}
            if filesPerSession > 1:
                p = d[mouseId][session][modelType]
                p['params'][fixedParamsIndex] = params[0]
                p['logLossTrain'][fixedParamsIndex] = logLossTrain[0]
                p['logLossTest'][fixedParamsIndex] = logLossTest[0]
                p['bicTrain'][fixedParamsIndex] = bicTrain[0]
                p['bicTest'][fixedParamsIndex] = bicTest[0]
        

## get experiment data and model variables
nSim = 10
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
            d[mouse][session]['Naive'] = {'logLossTest': sklearn.metrics.log_loss(obj.trialResponse,naivePrediction),
                                          'bicTest': 2 * sklearn.metrics.log_loss(obj.trialResponse,naivePrediction,normalize=False)}
            for modelType in modelTypes:
                if modelType not in d[mouse][session]:
                    continue
                s = d[mouse][session][modelType]
                s['pContext'] = []
                s['qReinforcement'] = []
                s['qPerseveration'] = []
                s['qReward'] = []
                s['qTotal'] = []
                s['prediction'] = []
                s['simulation'] = []
                s['simAction'] = []
                s['simPcontext'] = []
                s['simQreinforcement'] = []
                s['simQperseveration'] = []
                s['logLossSimulation'] = []                   
                for i,params in enumerate(s['params']):
                    pContext,qReinforcement,qPerseveration,qResp,qReward,qTotal,pAction,action = [val[0] for val in runModel(obj,*params,**s['paramsDict'])]
                    s['pContext'].append(pContext)
                    s['qReinforcement'].append(qReinforcement)
                    s['qPerseveration'].append(qPerseveration)
                    s['qReward'].append(qReward)
                    s['qTotal'].append(qTotal)
                    s['prediction'].append(pAction)
                    pContext,qReinforcement,qPerseveration,qResp,qReward,qTotal,pAction,action = runModel(obj,*params,useChoiceHistory=False,nReps=nSim,**s['paramsDict'])
                    s['simulation'].append(np.mean(pAction,axis=0))
                    s['simAction'].append(action)
                    s['simPcontext'].append(pContext)
                    s['simQreinforcement'].append(qReinforcement)
                    s['simQperseveration'].append(qPerseveration)
                    s['logLossSimulation'].append(np.mean([sklearn.metrics.log_loss(obj.trialResponse,p) for p in pAction]))


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
    d = np.concatenate(d).T[:,[0,1,3]]
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
for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
    d = [[[performanceData[trainingPhase][modelType][prm][lbl] for lbl in ('respFirst','respLast')] for prm in ('mice','Full model')] for modelType in modelTypes]
    d = np.concatenate(d).T[:,:,[0,1,3]]
    d = d[:,1] - d[:,0]
    mean = np.mean(d,axis=0)
    sem = np.std(d,axis=0)/(len(d)**0.5)
    ax.plot(x,mean,'o',mec=clr,mfc='none',ms=8,label=trainingPhase)
    for xi,m,s in zip(x,mean,sem):
        ax.plot([xi,xi],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(x)
ax.set_xticklabels(('Mice','Basic RL model','Context RL model')) #+modelTypes)
ax.set_yticks(np.arange(-0.1,1,0.1))
ax.set_xlim([-0.25,x[-1]+0.25])
ax.set_ylim([-0.1,0.4])
ax.set_ylabel('$\Delta$ Response probability to non-rewarded target\n(last trial previous block - first trial new block)',fontsize=12)
ax.legend(loc='upper center',fontsize=12)
plt.tight_layout()

for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(len(fixedParamNames[modelType])+1)
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
    ax.set_xticklabels(('Mice',)+fixedParamNames[modelType])
    ax.set_xlim([-0.25,len(x)+0.25])
    ax.set_ylim([0,3])
    ax.set_ylabel('Cross-modal d\'')
    ax.legend()
    plt.tight_layout()
    
for modelType in modelTypes:
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(len(fixedParamNames[modelType])+1)
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
    ax.set_xticklabels(('Mice',)+fixedParamNames[modelType])
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
    fig = plt.figure(figsize=((10 if modelType=='BasicRL' else 12),4))
    ax = fig.add_subplot(1,1,1)
    xticks = np.arange(len(fixedParamNames[modelType]))
    xlim = [-0.25,xticks[-1]+0.25]
    ax.plot(xlim,[0,0],'k--')
    for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
        d = modelData[trainingPhase]
        lh = np.array([np.mean([np.exp(-np.array(session[modelType]['logLossTest'])) for session in mouse.values() if modelType in session],axis=0) for mouse in d.values()])
        # lh = [np.mean([np.exp(-np.array(session[modelType]['logLossTest'])) for session in mouse.values() if modelType in session],axis=0) for mouse in d.values()]
        # lh = np.array([a for a in lh if len(a.shape)>0])
        lh -= lh[:,0][:,None]
        mean = np.mean(lh,axis=0)
        sem = np.std(lh,axis=0)/(len(lh)**0.5)
        x = np.arange(len(mean))
        ax.plot(x,mean,'o',mec=clr,mfc='none',ms=12,mew=2,alpha=0.75,label=trainingPhase)
        for xi,m,s in zip(x,mean,sem):
            ax.plot([xi,xi],[m-s,m+s],color=clr,alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(xticks)
    ax.set_xticklabels([s.replace(" ", "\n") for s in fixedParamNames[modelType]])
    ax.set_xlim(xlim)
    ax.set_ylim([-0.035,0.005])
    # ax.set_ylim([-0.005,0.02])
    ax.set_ylabel('$\Delta$ model likelihood',fontsize=14)
    # ax.set_title(modelType,fontsize=14)
    # ax.legend(loc=('upper left' if modelType=='BasicRL' else 'lower right'),fontsize=12)
    ax.legend(loc='lower left',fontsize=12)
    plt.tight_layout()
    
for modelType in modelTypes:
    fig = plt.figure(figsize=((10 if modelType=='BasicRL' else 12),4))
    ax = fig.add_subplot(1,1,1)
    xticks = np.arange(len(fixedParamNames[modelType]))
    xlim = [-0.25,xticks[-1]+0.25]
    ax.plot(xlim,[0,0],'k--')
    for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
        d = modelData[trainingPhase]
        lh = np.array([np.mean([session[modelType]['bicTest'] for session in mouse.values() if modelType in session],axis=0) for mouse in d.values()])
        lh -= lh[:,0][:,None]
        mean = np.mean(lh,axis=0)
        sem = np.std(lh,axis=0)/(len(lh)**0.5)
        x = np.arange(len(mean))
        ax.plot(x,mean,'o',mec=clr,mfc='none',ms=12,mew=2,alpha=0.75,label=trainingPhase)
        for xi,m,s in zip(x,mean,sem):
            ax.plot([xi,xi],[m-s,m+s],color=clr,lw=2,alpha=0.75)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(xticks)
    ax.set_xticklabels([s.replace(" ", "\n") for s in fixedParamNames[modelType]])
    ax.set_xlim(xlim)
    ax.set_ylim([-5,45])
    # ax.set_ylim([-10,20])
    ax.invert_yaxis()
    ax.set_ylabel('$\Delta$ BIC',fontsize=14)
    # ax.set_title(modelType,fontsize=14)
    # ax.legend(loc='upper left',fontsize=14)
    plt.tight_layout()

alim = (0,1)
for trainingPhase in trainingPhases:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alim,alim,'--',color='0.5')
    d = modelData[trainingPhase]
    x,y = [np.array([np.mean([np.exp(-np.array(session[modelType]['logLossTest'])) for session in mouse.values()],axis=0) for mouse in d.values()])[:,0] for modelType in modelTypes]
    ax.plot(x,y,'o',mec='k',mfc='none',alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Basic RL model',fontsize=16)
    ax.set_ylabel('Context RL model',fontsize=16)
    ax.set_title('likelihood',fontsize=18)
    plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,0],[0,1],'--',color='0.5')
for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
    d = modelData[trainingPhase]
    x,y = [np.array([np.mean([np.exp(-np.array(session[modelType]['logLossTest'])) for session in mouse.values()],axis=0) for mouse in d.values()])[:,0] for modelType in modelTypes]
    dsort = np.sort(y-x)
    cumProb = np.array([np.sum(dsort<=s)/dsort.size for s in dsort])
    ax.plot(dsort,cumProb,color=clr,label=trainingPhase)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_ylim([0,1.01])
ax.set_xlabel('$\Delta$ Likelihood (Context RL - Basic RL)',fontsize=16)
ax.set_ylabel('Cumulative fraction of mice',fontsize=16)
ax.legend()
plt.tight_layout()

alim = (0,1200)
for trainingPhase in trainingPhases:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alim,alim,'--',color='0.5')
    d = modelData[trainingPhase]
    x,y = [np.array([np.mean([session[modelType]['bicTest'] for session in mouse.values()],axis=0) for mouse in d.values()])[:,0] for modelType in modelTypes]
    ax.plot(x,y,'o',mec='k',mfc='none',alpha=0.5)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(np.arange(0,2000,500))
    ax.set_yticks(np.arange(0,2000,500))
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('Basic RL model',fontsize=16)
    ax.set_ylabel('Context RL model',fontsize=16)
    ax.set_title('BIC',fontsize=18)
    plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,0],[0,1],'--',color='0.5')
for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
    d = modelData[trainingPhase]
    x,y = [np.array([np.mean([session[modelType]['bicTest'] for session in mouse.values()],axis=0) for mouse in d.values()])[:,0] for modelType in modelTypes]
    dsort = np.sort(y-x)
    cumProb = np.array([np.sum(dsort<=s)/dsort.size for s in dsort])
    ax.plot(dsort,cumProb,color=clr,label=trainingPhase)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([-200,50])
ax.set_ylim([0,1.01])
ax.set_xlabel('$\Delta$ BIC (Context RL - Basic RL)',fontsize=16)
ax.set_ylabel('Cumulative fraction of mice',fontsize=16)
ax.legend()
plt.tight_layout()

            
    
## plot param values for each session
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
                    paramVals = np.array([session[modelType]['params'][i][prmInd] for mouse in d.values() for session in mouse.values() if modelType in session])
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
    
# correlations between parameters
for modelType in ('ContextRL',): #modelTypes:
    prms = [prm for prm in paramNames[modelType] if prm not in coreFixedPrms or prm in ['wPerseveration','alphaPerseveration','tauPerseveration']]
    for trainingPhase in ('after learning',): #trainingPhases:
        fig = plt.figure(figsize=(20,10))
        gs = matplotlib.gridspec.GridSpec(len(prms),len(prms))
        for i,yprm in enumerate(prms):
            for j,xprm in enumerate(prms):
                ax = fig.add_subplot(gs[i,j])
                d = modelData[trainingPhase]
                if len(d) > 0:
                    prmInd = [list(modelParams.keys()).index(prm) for prm in (xprm,yprm)]
                    paramVals = np.array([session[modelType]['params'][fixedParamNames[modelType].index('+perseveration')][prmInd] for mouse in d.values() for session in mouse.values() if modelType in session])
                    ax.plot(paramVals[:,0],paramVals[:,1],'ko',alpha=0.2)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=8)
                if i==0:
                    ax.set_title(xprm,fontsize=8)
                if j==0:
                    ax.set_ylabel(yprm,fontsize=8)
        plt.tight_layout()
        


## compare model and mice
var = 'simulation'
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for modelType in ('ContextRL',): #modelTypes:
    for phase in ('after learning',): #trainingPhases:
        for fixedParam in ('mice',) + fixedParamNames[modelType]:
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
                            if (phase=='odd' and blockInd+1 in (3,5)) or (phase=='even' and blockInd+1 in (2,4,6)) or (phase not in ('odd','even') and blockInd > 0):
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
        fig = plt.figure(figsize=(10,10))
        nRows = int(np.ceil((len(fixedParamNames[modelType])+1)/2))
        gs = matplotlib.gridspec.GridSpec(nRows,2)
        row = 0
        col = 0
        for fixedParam in ('mice',)+fixedParamNames[modelType]:
            ax = fig.add_subplot(gs[row,col])
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
                # y = [a for a in y if len(a.shape)>0]
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
            if row==nRows-1:
                ax.set_xticklabels([-5,-1,1,5,10,15])
                ax.set_xlabel('Trials after block switch',fontsize=14)
            else:
                ax.set_xticklabels([])
            ax.set_yticks([0,0.5,1])
            if col==0:
                ax.set_ylabel('Response rate',fontsize=14)
            else:
                ax.set_yticklabels([])
            ax.set_xlim([-preTrials-0.5,postTrials-0.5])
            ax.set_ylim([0,1.01])
            ax.set_title(fixedParam)
            #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
            plt.tight_layout()
            if row == nRows - 1:
                row = 0
                col += 1
            else:
                row += 1

    

## time dependence of effect of prior reward or response
stimType = ('rewarded target','non-rewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
prevTrialTypes = ('response to rewarded target','response to non-rewarded target')
modTypes = ('mice',) + modelTypes
fxdPrms = copy.deepcopy(fixedParamNames)
fxdPrms['mice'] = (None,)
blockEpochs = ('full',)
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
                                rewTrials = np.intersect1d(blockTrials,np.where(r & (obj.trialStim==rewStim))[0])
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
                                            anyTargetTrials = np.array([np.any(np.isin(obj.trialStim[p+1:s],(rewStim,otherModalTarget))) for s,p in zip(stimTrials,prevRespTrial)])
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

timeBins = np.array([0,5,10,15,20,30,40,60,100]) # [0,5,10,15,20,30,40,50,60,80,100] # [0,5,10,15,20,30,40,60,100]
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
            # fig.suptitle(modelType+', '+phase)
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
                    for d,r in zip(timeSince[mt][fp][phase][epoch][prevTrialType][stim],respNorm[mt][fp][phase][epoch][stim]):
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
                ax.tick_params(direction='out',top=False,right=False,labelsize=14)
                ax.set_xlim([0,90])
                ax.set_ylim([-0.1,0.15])
                if row == 0 and col == 1:
                    ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=16)
                if row == 0 and col == 1:
                    ax.set_ylabel('Response rate to '+stim+'\n(difference from within-block mean)',fontsize=16)
                ax.set_title(fixedParam,fontsize=16)
            plt.tight_layout()

stim = 'non-rewarded target'
for prevTrialType in prevTrialTypes: 
    for modelType in modelTypes:
        for phase in trainingPhases:
            fig = plt.figure(figsize=(12,10))
            # fig.suptitle(modelType+', '+phase)
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
                    for d,r in zip(timeSince[mt][fp][phase][epoch][prevTrialType][stim],respNorm[mt][fp][phase][epoch][stim]):
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
                ax.tick_params(direction='out',top=False,right=False,labelsize=14)
                ax.set_xlim([0,90])
                ax.set_ylim([0.2,0.7])
                if row == 0 and col == 1:
                    ax.set_xlabel('Time since last '+prevTrialType+' (s)',fontsize=16)
                if row == 0 and col == 1:
                    ax.set_ylabel('Response rate to '+stim,fontsize=16)
                ax.set_title(fixedParam,fontsize=16)
            plt.tight_layout()

trialBins = np.arange(20)
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
                    for d,r in zip(trialsSince[mt][fp][phase][epoch][prevTrialType][stim],respNorm[mt][fp][phase][epoch][stim]):
                        n.append(np.full(trialBins.size,np.nan))
                        p.append(np.full(trialBins.size,np.nan))
                        for i in trialBins:
                            j = d==i
                            n[-1][i] = j.sum()
                            p[-1][i] = r[j].sum() / n[-1][i]
                    m = np.nanmean(p,axis=0)
                    s = np.nanstd(p,axis=0) / (len(p)**0.5)
                    ax.plot(trialBins,m,color=clr,label=epoch)
                    ax.fill_between(trialBins,m-s,m+s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_ylim([-0.1,0.15])
                if row == 0 and col == 1:
                    ax.set_xlabel('Trials since last '+prevTrialType+' (s)',fontsize=14)
                if row == 2 and col == 0:
                    ax.set_ylabel('Response rate to '+stim,fontsize=14)
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
                    for d,r in zip(trialsSince[mt][fp][phase][epoch][prevTrialType][stim],resp[mt][fp][phase][epoch][stim]):
                        n.append(np.full(trialBins.size,np.nan))
                        p.append(np.full(trialBins.size,np.nan))
                        for i in trialBins:
                            j = d==i
                            n[-1][i] = j.sum()
                            p[-1][i] = r[j].sum() / n[-1][i]
                    m = np.nanmean(p,axis=0)
                    s = np.nanstd(p,axis=0) / (len(p)**0.5)
                    ax.plot(trialBins,m,color=clr,label=epoch)
                    ax.fill_between(trialBins,m-s,m+s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=12)
                ax.set_ylim([0.2,0.8])
                if row == 0 and col == 1:
                    ax.set_xlabel('Trials since last '+prevTrialType+' (s)',fontsize=14)
                if row == 2 and col == 0:
                    ax.set_ylabel('Response rate to '+stim,fontsize=14)
                ax.set_title(fixedParam)
            plt.tight_layout()
                    


## effect of prior reward or response
respType = 'response'
prevTrialTypes = ('response to rewarded target','response to non-rewarded target','response to non-target (rewarded modality)','response to non-target (unrewarded modality)')
stimTypes = ('rewarded target','non-rewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
respNext = {modelType: {prm: {phase: {prevTrialType: {stim: [] for stim in stimTypes} for prevTrialType in prevTrialTypes} for phase in trainingPhases} for prm in ('mice',)+fixedParamNames[modelType]} for modelType in modelTypes}
respPrev = copy.deepcopy(respNext)
respPrevNoRew = copy.deepcopy(respNext)
respMean = copy.deepcopy(respNext)
for modelType in modelTypes:
    for prm in respNext[modelType]:
        for phase in trainingPhases:
            for prevTrialType in prevTrialTypes:
                for s in stimTypes:
                    for mouse in modelData[phase]:
                        rn = []
                        rp = []
                        rpnr = []
                        rm = []
                        for session in modelData[phase][mouse]:
                            obj = sessionData[phase][mouse][session]
                            if prm=='mice': 
                                trialResponse = [obj.trialResponse]
                            elif prm in driftParams:
                                trialResponse = modelData[phase][mouse][session][modelType]['driftSimulation'][prm]['simAction']
                            else:    
                                trialResponse = modelData[phase][mouse][session][modelType]['simAction'][fixedParamNames[modelType].index(prm)]
                            for tr in trialResponse:
                                tr = tr.astype(bool)
                                isRespType = tr if respType=='response' else ~tr
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
                                    blockTrials = ~obj.autoRewardScheduled & (obj.trialBlock==blockInd+1)
                                    trials = np.where(blockTrials & (obj.trialStim==stim))[0]
                                    if prevTrialType == 'response to rewarded target':
                                        prevInd = isRespType & (obj.trialStim == rewStim) & ~obj.autoRewardScheduled
                                    elif prevTrialType == 'response to non-rewarded target':
                                        prevInd = isRespType & (obj.trialStim == nonRewStim)
                                    elif prevTrialType == 'response to non-target (rewarded modality)':
                                        prevInd = isRespType & (obj.trialStim == rewStim[:-1]+'2')
                                    elif prevTrialType == 'response to non-target (unrewarded modality)':
                                        prevInd = isRespType & (obj.trialStim == nonRewStim[:-1]+'2')
                                    rn.append(tr[trials][prevInd[trials-1]])
                                    prevInd = np.concatenate((prevInd,[False]))
                                    rp.append(tr[trials][prevInd[trials+1]])
                                    rpnr.append(tr[trials][prevInd[trials+1] & ~(tr & (obj.trialStim==rewStim))[trials-1]])
                                    rm.append(np.mean(tr[trials]))
                        if len(rn) > 0:
                            rn = np.concatenate(rn)
                            rp = np.concatenate(rp)
                            rpnr = np.concatenate(rpnr)
                        respNext[modelType][prm][phase][prevTrialType][s].append(np.nanmean(rn))
                        respPrev[modelType][prm][phase][prevTrialType][s].append(np.nanmean(rp))
                        respPrevNoRew[modelType][prm][phase][prevTrialType][s].append(np.nanmean(rpnr))
                        respMean[modelType][prm][phase][prevTrialType][s].append(np.nanmean(rm))

stimLabels = ('rewarded target','non-rewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')

alim = (0,1.02)
for modelType in ('ContextRL',): #modTypes:
    for fixedParam in ('Full model',): #fxdPrms[modelType]:
        for d,lbl in zip((respMean,respPrev,respPrevNoRew),('(within block mean)','trial t-1','trial t-1 (no reward t-2)')):
            for phase in ('after learning',): #trainingPhases:
                for prevTrialType in prevTrialTypes:
                    fig = plt.figure(figsize=(10,8))
                    ax = fig.add_subplot(1,1,1)
                    ax.plot(alim,alim,'k--')
                    for stim,mec,mfc in zip(stimTypes,'gmgm',('g','m','none','none')):
                        ax.plot(d[modelType][fixedParam][phase][prevTrialType][stim],respNext[modelType][fixedParam][phase][prevTrialType][stim],'o',mec=mec,mfc=mfc,ms=10,label=stim)
                    for side in ('right','top'):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
                    ax.set_xticks([0,0.5,1])
                    ax.set_yticks([0,0.5,1])
                    ax.set_xlim(alim)
                    ax.set_ylim(alim)
                    ax.set_aspect('equal')
                    ax.set_xlabel('Response prob '+lbl,fontsize=20)
                    ax.set_ylabel('Response prob trial t+1',fontsize=20)
                    ax.set_title('Conditioned on trial t '+prevTrialType,fontsize=20)
                    # ax.legend(loc='lower right',fontsize=12)
                    plt.tight_layout()

cmax = 0.35
for modelType in modelTypes:
    for phase in ('after learning',):
        for prm in respNext[modelType]:
            for d,lbl in zip((respMean,respPrev,respPrevNoRew),('within block mean','response prob trial t-1','response prob trial t-1 (no reward t-2)')):   
                r = np.full((len(stimTypes),len(prevTrialTypes)),np.nan)    
                for i,stim in enumerate(stimTypes):
                    for j,prevTrialType in enumerate(prevTrialTypes):
                        r[i,j] = np.nanmean(np.array(respNext[modelType][prm][phase][prevTrialType][stim]) - np.array(d[modelType][prm][phase][prevTrialType][stim]))
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(1,1,1)
                im = ax.imshow(r,cmap='bwr',clim=(-cmax,cmax))
                cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.04)
                # cb.set_ticks([-1,-0.5,0,0.5,1])
                for side in ('right','top','left','bottom'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=10)
                ax.set_xticks(np.arange(4))
                ax.set_xticklabels(stimLabels)
                ax.set_yticks(np.arange(4))
                ax.set_yticklabels(stimLabels)
                ax.set_xlabel('Response to stimulus on trial t',fontsize=16)
                ax.set_ylabel('Response prob trial t+1\nminus '+lbl,fontsize=16)
                ax.set_title('Change in response probability',fontsize=16)

                    
## intra-block resp correlations
params = ('mice',) + fixedParamNames['ContextRL']
phases = ('after learning',)
autoCorrMat = {modelType: {prm: {phase: np.zeros((4,len(modelData[phase]),100)) for phase in phases} for prm in params} for modelType in modelTypes}
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
corrWithinMat = {modelType: {prm: {phase: np.zeros((4,4,len(modelData[phase]),200)) for phase in phases} for prm in params} for modelType in modelTypes}
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
for modelType in ('ContextRL',):#modelTypes:
    for prm in autoCorrMat[modelType]:
        for phase in autoCorrMat[modelType][prm]:
            for m,mouse in enumerate(modelData[phase]):
                autoCorr = []
                autoCorrDetrend = []
                corrWithin = []
                corrWithinDetrend = []
                    
                for session in modelData[phase][mouse]:
                    obj = sessionData[phase][mouse][session]   
                    if prm=='mice': 
                        trialResp = [obj.trialResponse]
                    else:    
                        trialResp = modelData[phase][mouse][session][modelType]['simAction'][fixedParamNames[modelType].index(prm)]
                    rr,ac,acr,acd,cw,cwr,cwd,ca= getResponseCorrelations(obj,trialResp)
                    autoCorr.append(ac)
                    autoCorrDetrend.append(acd)
                    corrWithin.append(cw)
                    corrWithinDetrend.append(cwd)
                      
                autoCorrMat[modelType][prm][phase][:,m] = np.nanmean(autoCorr,axis=(0,2))
                autoCorrDetrendMat[modelType][prm][phase][:,m] = np.nanmean(autoCorrDetrend,axis=(0,2))
                    
                corrWithinMat[modelType][prm][phase][:,:,m] = np.nanmean(corrWithin,axis=(0,3))
                corrWithinDetrendMat[modelType][prm][phase][:,:,m] = np.nanmean(corrWithinDetrend,axis=(0,3))

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')

modelType = 'ContextRL'        
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
for prm in list(corrWithinMat[modelType].keys())[1:]:
    fig = plt.figure(figsize=(12,10))
    # fig.suptitle(fixedParam)         
    gs = matplotlib.gridspec.GridSpec(4,4)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:4]):
            ax = fig.add_subplot(gs[i,j])
            for lbl,clr in zip(('mice',prm),'kr'):
                mat = corrWithinDetrendMat[modelType][lbl][phase][i,j,:,1:]
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
    
modelType = 'ContextRL'        
phase = 'after learning'
for prm in list(corrWithinMat[modelType].keys())[1:]:
    fig = plt.figure(figsize=(6,10))
    # fig.suptitle(fixedParam)         
    gs = matplotlib.gridspec.GridSpec(4,2)
    x = np.arange(1,200)
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels[:2]):
            ax = fig.add_subplot(gs[i,j])
            for lbl,clr in zip(('mice',prm),'kr'):
                mat = corrWithinDetrendMat[modelType][lbl][phase][i,j,:,1:]
                m = np.nanmean(mat,axis=0)
                s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
                ax.plot(x,m,clr,label=lbl)
                ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=12)
            ax.set_xlim([0,20])
            ax.set_ylim([-0.035,0.045])
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
    plt.tight_layout()
                    


