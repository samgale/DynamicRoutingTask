#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:47:48 2023

@author: samgale
"""

import glob
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn.metrics
from DynamicRoutingAnalysisUtils import getSessionData, calcDprime
from RLmodelHPC import calcLogisticProb, runModel


# plot relationship bewtween tau and q values
q = np.arange(0,1.01,0.01)
beta = np.arange(51)
bias = (0,0.25)
xticks = np.arange(0,q.size+1,int(q.size/4))
yticks = np.arange(0,50,10)
for bi in bias:
    p = np.zeros((beta.size,q.size))
    for i,bt in enumerate(beta):
        p[i] = calcLogisticProb(q,bt,bi)
    
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
        ax.plot(q,calcLogisticProb(q,bt,bi),color=clr,ls=ls,label=r'$\beta$='+str(bt)+', bias='+str(bi))
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


# toy example comparing logloss, dprime, and resp rate change
n = 15
x = np.arange(n)
respA = [np.ones(n),np.linspace(0.5,1,n),np.zeros(n)+0.76]
respB = [np.zeros(n)+0.5,np.linspace(1,0.5,n),np.zeros(n)+0.74]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
logLoss = []
dprime = []
for i,(a,b) in enumerate(zip(respA,respB)):
    ax.plot(x,a,'g')
    ax.plot(x,b,'m')
    if i>0:
        logLoss.append(sklearn.metrics.log_loss([1,1,0,1],[a.mean()]*2+[b.mean()]*2))
    dprime.append(calcDprime(a.mean(),b.mean(),n,n))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
plt.tight_layout()


# get fit params from HPC output
trainingPhaseNames = ('initial training','after learning')#,'nogo','noAR','rewardOnly','no reward')
trainingPhaseColors = 'mgrbck'
modelTypeNames = ('basicRL','contextRLmultiState','contextRLmultiStateRPE','contextRLweightStates','contextRLweightStatesRPE')
modelTypeColors = 'krmbc'

paramNames = {}
paramBounds = {}
fixedParamNames = {}
fixedParamValues = {}
nModelParams = {}
for modelType in modelTypeNames:
    if modelType == 'basicRL':
        paramNames[modelType] = ('betaAction','biasAction','biasAttention','visConf','audConf','alphaReward','alphaStim')
        paramBounds[modelType] = ([0,40],[-1,1],[-1,1],[0.5,1],[0.5,1],[0,1],[0,1])
        fixedParamNames[modelType] = ('Full model','biasAction','biasAttention','visConf','audConf','alphaReward','alphaStim')
        fixedParamValues[modelType] = (None,0,0,1,1,0,0)
        nModelParams[modelType] = (6,5,5,5,5,5,5)
    else:
        paramNames[modelType] = ('betaAction','biasAction','biasAttention','visConf','audConf','alphaReward','alphaStim','alphaContext','decayContext','alphaHabit')
        paramBounds[modelType] = ([0,40],[-1,1],[-1,1],[0.5,1],[0.5,1],[0,1],[0,1],[0,1],[1,600],[0,1])
        fixedParamNames[modelType] = ('Full model','biasAction','biasAttention','visConf','audConf','alphaReward','alphaStim','alphaContext','alphaContext,\nalphaStim','decayContext','alphaHabit','decayContext,\nalphaHabit')
        fixedParamValues[modelType] = (None,0,0,1,1,0,0,0,0,0,0,0)
        nModelParams[modelType] = (9,8,8,8,8,8,8,7,6,8,8,7)

trainingPhases = []
modelTypes = []
modelTypeParams = {}
modelData = {}
filePaths = glob.glob(os.path.join(r"\\allen\programs\mindscope\workgroups\dynamicrouting\Sam\RLmodel",'*.npz'))
for f in filePaths:
    mouseId,sessionDate,sessionTime,trainingPhase,modelType = os.path.splitext(os.path.basename(f))[0].split('_')
    session = sessionDate+'_'+sessionTime
    if trainingPhase not in trainingPhases:
        trainingPhases.append(trainingPhase)
        modelData[trainingPhase] = {}
    if modelType not in modelTypes:
        modelTypes.append(modelType)
    with np.load(f) as data:
        params = data['params']
        logLoss = data['logLoss']
        termMessage = data['terminationMessage']
        if modelType not in modelTypeParams:
            modelTypeParams[modelType] = {key: bool(val) for key,val in data.items() if key not in ('params','logLoss','terminationMessage')}
    d = modelData[trainingPhase]
    p = {'params': params, 'logLossTrain': logLoss, 'terminationMessage': termMessage}
    if mouseId not in d:
        d[mouseId] = {session: {modelType: p}}
    elif session not in d[mouseId]:
        d[mouseId][session] = {modelType: p}
    elif modelType not in d[mouseId][session]:
        d[mouseId][session][modelType] = p
trainingPhases = [name for name in trainingPhaseNames if name in trainingPhases]
modelTypes = [name for name in modelTypeNames if name in modelTypes]


# print fit termination message
for trainingPhase in trainingPhases:
    for mouse in modelData[trainingPhase]:
        for session in modelData[trainingPhase][mouse]:
            for modelType in modelTypes:
                print(modelData[trainingPhase][mouse][session][modelType]['terminationMessage'])


# get experiment data and model variables
sessionData = {phase: {} for phase in trainingPhases}
useHistory = True
nReps = 1      
for trainingPhase in trainingPhases:
    print(trainingPhase)
    d = modelData[trainingPhase]
    if len(d) > 0:
        for mouse in d:
            for session in d[mouse]:
                obj = getSessionData(mouse,session)
                if mouse not in sessionData[trainingPhase]:
                    sessionData[trainingPhase][mouse] = {session: obj}
                elif session not in sessionData[trainingPhase][mouse]:
                    sessionData[trainingPhase][mouse][session] = obj
                naivePrediction = np.zeros(obj.nTrials) + (obj.trialResponse.sum() / obj.trialResponse.size)
                d[mouse][session]['Naive'] = {'logLossTest': sklearn.metrics.log_loss(obj.trialResponse,naivePrediction)}
                for modelType in modelTypes:
                    s = d[mouse][session][modelType]
                    s['pContext'] = []
                    s['qContext'] = []
                    s['qStim'] = []
                    s['wHabit'] = []
                    s['wReward'] = []
                    s['expectedValue'] = []
                    s['qTotal'] = []
                    s['prediction'] = []
                    s['logLossTest'] = []
                    for params in s['params']:
                        pContext,qContext,qStim,wHabit,wReward,expectedValue,qTotal,pAction,action = [val.mean(axis=0) for val in runModel(obj,*params,useHistory=useHistory,nReps=nReps,**modelTypeParams[modelType])]
                        s['pContext'].append(pContext)
                        s['qContext'].append(qContext)
                        s['qStim'].append(qStim)
                        s['wHabit'].append(wHabit)
                        s['wReward'].append(wReward)
                        s['expectedValue'].append(expectedValue)
                        s['qTotal'].append(qTotal)
                        s['prediction'].append(pAction)
                        s['logLossTest'].append(sklearn.metrics.log_loss(obj.trialResponse,pAction))

                        
# get performance data
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
                        resp = d[mouse][session][modelType]['prediction'][fixedParamNames[modelType].index(fixedParam)]
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


# plot performance data
for modelType in ('basicRL',):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    xlbls = ('mice',) + fixedParamNames[modelType]
    for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
        for x,lbl in enumerate(xlbls):
            d = performanceData[trainingPhase][modelType][lbl]['dprime']
            m = np.mean(d)
            s = np.std(d)/(len(d)**0.5)
            lbl = trainingPhase if x==0 else None
            ax.plot(x,m,'o',mec=clr,mfc='none',ms=10,mew=2,label=lbl)
            ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(len(xlbls)))
    ax.set_xticklabels(['mice','full model']+[name+'='+str(val) for name,val in zip(fixedParamNames[modelType][1:],fixedParamValues[modelType][1:])])
    ax.set_xlim([-0.25,len(xlbls)+0.25])
    # ax.set_ylim([0,0.7])
    ax.set_ylabel('cross-modal d\'')
    ax.set_title(modelType)
    ax.legend(loc='upper right')
    plt.tight_layout()
    
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,1,1)
xlbls = ('mice',) + fixedParamNames['contextRLmultiState']
for x,lbl in enumerate(xlbls):
    for i,(modelType,clr) in enumerate(zip(modelTypes[1:],modelTypeColors[1:])):
        if lbl=='mice':
            if i>0:
                continue
            c = 'k'
            lb = 'mice'
        else:
            c = clr
            lb = modelType if x==1 else None
        d = performanceData['after learning'][modelType][lbl]['dprime']
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax.plot(x,m,'o',mec=c,mfc='none',ms=10,mew=2,label=lb)
        ax.plot([x,x],[m-s,m+s],color=c,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(len(xlbls)))
ax.set_xticklabels(['mice','full model']+[name+'='+str(val) for name,val in zip(fixedParamNames[modelType][1:],fixedParamValues[modelType][1:])])
ax.set_xlim([-0.25,len(xlbls)+0.25])
# ax.set_ylim([0,0.7])
ax.set_ylabel('cross-modal d\'')
ax.set_title('after learning')
ax.legend(loc='lower left')
plt.tight_layout()
    
for modelType in ('basicRL',):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    xlbls = ('mice',) + fixedParamNames[modelType]
    ax.plot([-1,len(xlbls)+1],[0,0],'k--')
    for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
        for x,lbl in enumerate(xlbls):
            respFirst = performanceData[trainingPhase][modelType][lbl]['respFirst']
            respLast = performanceData[trainingPhase][modelType][lbl]['respLast']
            d = np.array(respFirst) - np.array(respLast)
            m = np.mean(d)
            s = np.std(d)/(len(d)**0.5)
            lbl = trainingPhase if x==0 else None
            ax.plot(x,m,'o',mec=clr,mfc='none',ms=10,mew=2,label=lbl)
            ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(len(xlbls)))
    ax.set_xticklabels(['mice','full model']+[name+'='+str(val) for name,val in zip(fixedParamNames[modelType][1:],fixedParamValues[modelType][1:])])
    ax.set_xlim([-0.25,len(xlbls)+0.25])
    # ax.set_ylim([0,0.7])
    ax.set_ylabel('$\Delta$ Response rate to non-rewarded target\n(first trial - last trial previous block)')
    ax.set_title(modelType)
    ax.legend(loc='lower right')
    plt.tight_layout()

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,1,1)
xlbls = ('mice',) + fixedParamNames['contextRLmultiState']
ax.plot([-1,len(xlbls)+1],[0,0],'k--')
for x,lbl in enumerate(xlbls):
    for i,(modelType,clr) in enumerate(zip(modelTypes[1:],modelTypeColors[1:])):
        if lbl=='mice':
            if i>0:
                continue
            c = 'k'
            lb = 'mice'
        else:
            c = clr
            lb = modelType if x==1 else None
        respFirst = performanceData[trainingPhase][modelType][lbl]['respFirst']
        respLast = performanceData[trainingPhase][modelType][lbl]['respLast']
        d = np.array(respFirst) - np.array(respLast)
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax.plot(x,m,'o',mec=c,mfc='none',ms=10,mew=2,label=lb)
        ax.plot([x,x],[m-s,m+s],color=c,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(len(xlbls)))
ax.set_xticklabels(['mice','full model']+[name+'='+str(val) for name,val in zip(fixedParamNames[modelType][1:],fixedParamValues[modelType][1:])])
ax.set_xlim([-0.25,len(xlbls)+0.25])
# ax.set_ylim([0,0.7])
ax.set_ylabel('$\Delta$ Response rate to non-rewarded target\n(first trial - last trial previous block)')
ax.set_title('after learning')
ax.legend(loc='upper left')
plt.tight_layout()


# plot logloss
fig = plt.figure(figsize=(14,4))
ax = fig.add_subplot(1,1,1)
xlbls = ['Naive'] + modelTypes
for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
    d = modelData[trainingPhase]
    if len(d) > 0:
        for x,xlbl in enumerate(xlbls):
            val = np.array([np.mean([session[xlbl]['logLossTest'] for session in mouse.values()],axis=0) for mouse in d.values()])
            if xlbl != 'Naive':
                val = val[:,fixedParamNames[xlbl].index('Full model')]
            m = val.mean()
            s = val.std()/(len(val)**0.5)
            lbl = trainingPhase if x==0 else None
            ax.plot(x,m,'o',mec=clr,mfc='none',ms=10,mew=2,label=lbl)
            ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(len(xlbls)))
ax.set_xticklabels(['Naive model\n(constant response probability)'] + modelTypes)
ax.set_xlim([-0.25,len(xlbls)+0.25])
ax.set_ylim([0,0.7])
ax.set_ylabel('Negative log-likelihood')
ax.legend(loc='lower right')
plt.tight_layout()

for modelType in ('basicRL',):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    xticks = np.arange(len(fixedParamNames[modelType]))
    xlim = [-0.25,xticks[-1]+0.25]
    ax.plot(xlim,[0,0],'--',color='0.5')
    for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
        d = modelData[trainingPhase]
        if len(d) > 0:
            val = np.array([np.mean([session[modelType]['logLossTest'] for session in mouse.values()],axis=0) for mouse in d.values()])
            val -= val[:,fixedParamNames[modelType].index('Full model')][:,None]
            mean = val.mean(axis=0)
            sem = val.std(axis=0)/(len(val)**0.5)
            ax.plot(xticks,mean,'o',mec=clr,mfc='none',ms=10,mew=2,label=trainingPhase)
            for x,m,s in zip(xticks,mean,sem):
                ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels([fixedParamNames[modelType][0]]+[name+'='+str(val) for name,val in zip(fixedParamNames[modelType][1:],fixedParamValues[modelType][1:])])
    ax.set_xlim(xlim)
    ax.set_ylabel(r'$\Delta$ NLL')
    ax.set_title(modelType)
    ax.legend(loc='upper left')
    plt.tight_layout()
    
fig = plt.figure(figsize=(14,4))
ax = fig.add_subplot(1,1,1)
xticks = np.arange(len(fixedParamNames['contextRLmultiState']))
xlim = [-0.25,xticks[-1]+0.25]
ax.plot(xlim,[0,0],'--',color='0.5')
for modelType,clr in zip(modelTypes[1:],modelTypeColors[1:]):
    d = modelData['after learning']
    if len(d) > 0:
        val = np.array([np.mean([session[modelType]['logLossTest'] for session in mouse.values()],axis=0) for mouse in d.values()])
        val -= val[:,fixedParamNames[modelType].index('Full model')][:,None]
        mean = val.mean(axis=0)
        sem = val.std(axis=0)/(len(val)**0.5)
        ax.plot(xticks,mean,'o',mec=clr,mfc='none',ms=10,mew=2,label=modelType)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xticklabels([fixedParamNames[modelType][0]]+[name+'='+str(val) for name,val in zip(fixedParamNames[modelType][1:],fixedParamValues[modelType][1:])])
ax.set_xlim(xlim)
ax.set_ylabel(r'$\Delta$ NLL')
ax.set_title('after learning')
ax.legend(loc='upper left')
plt.tight_layout()

for modelType in modelTypes:
    fig = plt.figure(figsize=(5,10))
    for i,(fixedParam,fixedVal) in enumerate(zip(fixedParamNames[modelType],fixedParamValues[modelType])):
        ax = fig.add_subplot(len(fixedParamNames[modelType]),1,i+1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
            d = modelData[trainingPhase]
            if len(d) > 0:
                logLoss = np.array([np.mean([session[modelType]['logLossTrain'] for session in mouse.values()],axis=0) for mouse in d.values()])          
                logLoss = logLoss[:,i] - logLoss[:,fixedParamNames[modelType].index('Full model')] if fixedParam != 'Full model' else logLoss[:,i]
                dsort = np.sort(logLoss)
                cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
                ax.plot(dsort,cumProb,color=clr,label=trainingPhase)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim(([0,1] if fixedParam == 'Full model' else [-0.05,0.2]))
        ax.set_ylim([0,1.01])
        ax.set_xlabel(('NLL' if fixedParam == 'Full model' else r'$\Delta$ NLL'))
        ax.set_title((fixedParam if fixedParam == 'Full model' else fixedParam+'='+str(fixedVal)))
        if i==0:
            ax.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
                
                
# plot param values
for modelType in ('basicRL',):
    fig = plt.figure(figsize=(11,11))
    gs = matplotlib.gridspec.GridSpec(len(fixedParamNames[modelType]),len(paramNames[modelType]))
    for i,(fixedParam,fixedVal) in enumerate(zip(fixedParamNames[modelType],fixedParamValues[modelType])):
        for j,(param,xlim) in enumerate(zip(paramNames[modelType],paramBounds[modelType])):
            ax = fig.add_subplot(gs[i,j])
            for trainingPhase,clr in zip(trainingPhases,trainingPhaseColors):
                d = modelData[trainingPhase]
                if len(d) > 0:
                    paramVals = np.array([np.mean([session[modelType]['params'][i,j] for session in mouse.values()]) for mouse in d.values()])
                    if len(np.unique(paramVals)) > 1:
                        dsort = np.sort(paramVals)
                        cumProb = np.array([np.sum(dsort<=s)/dsort.size for s in dsort])
                        ax.plot(dsort,cumProb,color=clr,label=trainingPhase)
                    else:
                        ax.plot(paramVals[0],1,'o',mfc=clr,mec=clr)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
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
                ax.set_title((fixedParam+'('+modelType+')' if fixedParam == 'Full model' else fixedParam+'='+str(fixedVal)))
            if i==0 and j==len(paramNames[modelType])-1:
                ax.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    
fig = plt.figure(figsize=(14,11))
modelType = 'contextRLmultiState'
gs = matplotlib.gridspec.GridSpec(len(fixedParamNames[modelType]),len(paramNames[modelType]))
for i,(fixedParam,fixedVal) in enumerate(zip(fixedParamNames[modelType],fixedParamValues[modelType])):
    for j,(param,xlim) in enumerate(zip(paramNames[modelType],paramBounds[modelType])):
        ax = fig.add_subplot(gs[i,j])
        for modelType,clr in zip(modelTypes[1:],modelTypeColors[1:]):
            d = modelData['after learning']
            if len(d) > 0:
                paramVals = np.array([np.mean([session[modelType]['params'][i,j] for session in mouse.values()]) for mouse in d.values()])
                if len(np.unique(paramVals)) > 1:
                    dsort = np.sort(paramVals)
                    cumProb = np.array([np.sum(dsort<=s)/dsort.size for s in dsort])
                    ax.plot(dsort,cumProb,color=clr,label=(modelType if i==0 and j==len(paramNames[modelType])-1 else None))
                else:
                    ax.plot(paramVals[0],1,'o',mfc=clr,mec=clr)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
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
            ax.set_title((fixedParam+'(after learning)' if fixedParam == 'Full model' else fixedParam+'='+str(fixedVal)))
        if i==0 and j==len(paramNames[modelType])-1:
            ax.legend(bbox_to_anchor=(1,1))
plt.tight_layout()


# compare model and mice
for modelType in modelTypes[1:]:
    var = 'prediction'
    stimNames = ('vis1','vis2','sound1','sound2')
    preTrials = 5
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)
    for trainingPhase in trainingPhases[1:]:
        fig = plt.figure(figsize=(12,10))
        nRows = int(np.ceil((len(fixedParamNames[modelType])+1)/2))
        gs = matplotlib.gridspec.GridSpec(nRows,4)
        for i,(fixedParam,fixedVal) in enumerate(zip(('mice',) + fixedParamNames[modelType],(None,)+fixedParamValues[modelType])):
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
                if modelType != 'basicRL' and col>1:
                    row += 1
                ax = fig.add_subplot(gs[row,col])
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
                                if rewStim==rewardStim and blockInd > 0:
                                    trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
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
                    title = fixedParam+'='+str(fixedVal)
                ax.set_title(title)
                if i==0 and j==1:
                    ax.legend(bbox_to_anchor=(1,1))
        plt.tight_layout()
            
preTrials = 0
postTrials = 15
x = np.arange(-preTrials,postTrials+1)
for var,yticks,ylim,ylbl in zip(('prediction','expectedValue'),([0,0.5,1],[-1,0,1]),([0,1.01],[-1.01,1.01]),('Response\nrate','Expected\nvalue')):
    if var=='expectedValue':
        continue
    for trainingPhase in trainingPhases:
        fig = plt.figure(figsize=(8,10))
        gs = matplotlib.gridspec.GridSpec(len(fixedParamNames)+1,2)
        for i,(fixedParam,fixedVal) in enumerate(zip(('mice',) + fixedParamNames,(None,)+fixedParamValues)):
            if fixedParam == 'mice':
                d = sessionData[trainingPhase]
            else:
                d = modelData[trainingPhase]
            if len(d) == 0:
                continue
            for j,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks'))):
                ax = fig.add_subplot(gs[i,j])
                for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                    y = []
                    for mouse in d:
                        y.append([])
                        for session in d[mouse]:
                            obj = sessionData[trainingPhase][mouse][session]
                            if fixedParam == 'mice':
                                resp = obj.trialResponse
                            else:
                                resp = d[mouse][session][modelType][var][fixedParamNames.index(fixedParam)]
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if rewStim==rewardStim and blockInd == 0:
                                    trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
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
        

# plot pContext and wHabit
modelType = 'contextQ'
preTrials = 20
postTrials = 60
x = np.arange(-preTrials,postTrials+1)
for var,ylbl in zip(('pContext','wHabit'),('Context belief','Habit weight')):
    for trainingPhase in trainingPhases:
        d = modelData[trainingPhase]
        if len(d) == 0:
            continue
        fig = plt.figure(figsize=(10,10))
        gs = matplotlib.gridspec.GridSpec(len(fixedParamNames),2)
        for i,(fixedParam,fixedVal) in enumerate(zip(fixedParamNames,fixedParamValues)):
            for j,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks'))):
                ax = fig.add_subplot(gs[i,j])
                contexts,clrs = (('visual','auditory'),'gm') if var=='pContext' else ((None,),'k')
                for contextInd,(context,clr) in enumerate(zip(contexts,clrs)):
                    y = []
                    for mouse in d:
                        y.append([])
                        for session in d[mouse]:
                            obj = sessionData[trainingPhase][mouse][session]
                            v = d[mouse][session][modelType][var][fixedParamNames.index(fixedParam)]
                            if var=='pContext':
                                v = v[:,contextInd]
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if rewStim==rewardStim and blockInd > 0:
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = v[(obj.trialBlock==blockInd)]
                                    k = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-k:preTrials] = pre[-k:]
                                    post = v[(obj.trialBlock==blockInd+1)]
                                    k = min(postTrials,post.size)
                                    y[-1][-1][preTrials+1:preTrials+1+k] = post[:k]
                        y[-1] = np.nanmean(y[-1],axis=0)
                    m = np.nanmean(y,axis=0)
                    s = np.nanstd(y,axis=0)/(len(y)**0.5)
                    ax.plot(x,m,color=clr,label=context)
                    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
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

preTrials = 0
postTrials = 60
x = np.arange(-preTrials,postTrials+1)
for var,ylbl in zip(('pContext','wHabit'),('Context belief','Habit weight')):
    for trainingPhase in trainingPhases:
        d = modelData[trainingPhase]
        if len(d) == 0:
            continue
        fig = plt.figure(figsize=(10,10))
        gs = matplotlib.gridspec.GridSpec(len(fixedParamNames),2)
        for i,(fixedParam,fixedVal) in enumerate(zip(fixedParamNames,fixedParamValues)):
            for j,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks'))):
                ax = fig.add_subplot(gs[i,j])
                contexts,clrs = (('visual','auditory'),'gm') if var=='pContext' else ((None,),'k')
                for contextInd,(context,clr) in enumerate(zip(contexts,clrs)):
                    y = []
                    for mouse in d:
                        y.append([])
                        for session in d[mouse]:
                            obj = sessionData[trainingPhase][mouse][session]
                            v = d[mouse][session][modelType][var][fixedParamNames.index(fixedParam)]
                            if var=='pContext':
                                v = v[:,contextInd]
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if rewStim==rewardStim and blockInd == 0:
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = v[(obj.trialBlock==blockInd)]
                                    k = min(preTrials,pre.size)
                                    y[-1][-1][preTrials-k:preTrials] = pre[-k:]
                                    post = v[(obj.trialBlock==blockInd+1)]
                                    k = min(postTrials,post.size)
                                    y[-1][-1][preTrials+1:preTrials+1+k] = post[:k]
                        y[-1] = np.nanmean(y[-1],axis=0)
                    m = np.nanmean(y,axis=0)
                    s = np.nanstd(y,axis=0)/(len(y)**0.5)
                    ax.plot(x,m,color=clr,label=context)
                    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
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
        
        
# plot q values
modelType = 'contextQ'
preTrials = 20
postTrials = 60
for trainingPhase in trainingPhases:
    d = modelData[trainingPhase]
    if len(d) == 0:
        continue
    for qval in ('qContext','qStim'):
        fig = plt.figure(figsize=(10,10))
        gs = matplotlib.gridspec.GridSpec(len(fixedParamNames),2)
        for i,(fixedParam,fixedVal) in enumerate(zip(fixedParamNames,fixedParamValues)):
            for j,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks'))):
                ax = fig.add_subplot(gs[i,j])
                y = []
                for mouse in d:
                    y.append([])
                    for session in d[mouse]:
                        obj = sessionData[trainingPhase][mouse][session]
                        q = d[mouse][session][modelType][qval][fixedParamNames.index(fixedParam)]
                        if qval=='qContext':
                            q = q.reshape(obj.nTrials,8)
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if rewStim==rewardStim and blockInd > 0:
                                y[-1].append(np.full((q.shape[1],preTrials+postTrials),np.nan))
                                pre = q[(obj.trialBlock==blockInd)]
                                k = min(preTrials,len(pre))
                                y[-1][-1][:,preTrials-k:preTrials] = pre[-k:].T
                                post = q[(obj.trialBlock==blockInd+1)]
                                k = min(postTrials,len(post))
                                y[-1][-1][:,preTrials:preTrials+k] = post[:k].T
                    y[-1] = np.nanmean(y[-1],axis=0)
                m = np.nanmean(y,axis=0)
                im = ax.imshow(m,clim=(0,1),cmap='gray')
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks(np.arange(0,preTrials+postTrials,20))
                ax.set_xticklabels(np.arange(-20,postTrials,20))
                # ax.set_yticks([0,0.5,1])
                ax.set_xlim([-0.5,preTrials+postTrials+0.5])
                # ax.set_ylim([0,1.01])
                if i==len(fixedParamNames)-1:
                    ax.set_xlabel('Trials after block switch')
                if j==0:
                    ax.set_ylabel('state')
                if fixedParam=='Full model':
                    title = fixedParam+', '+blockLabel
                else:
                    title = fixedParam+'='+str(fixedVal)
                ax.set_title(title)
                if i==0 and j==1:
                    cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
                    cb.ax.tick_params(length=0)
                    cb.set_ticks([0,0.5,1])
        plt.tight_layout()


# no reward blocks, target stimuli only
for modelType in modelTypes:
    fig = plt.figure(figsize=(8,10))
    gs = matplotlib.gridspec.GridSpec(len(fixedParamNames)+1,2)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)  
    for i,(fixedParam,fixedVal) in enumerate(zip(('mice',) + fixedParamNames,(None,)+fixedParamValues)):
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
                            resp = d[mouse][session][modelType]['prediction'][fixedParamNames.index(fixedParam)]
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



