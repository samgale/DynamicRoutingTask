#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:47:48 2023

@author: samgale
"""

import copy
import glob
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getSessionData
from RLmodelHPC import calcLogisticProb, runModel


# plot relationship bewtween tau and q values
q = np.arange(-1,1.01,0.01)
tau = np.arange(0.01,2.01,0.01)
bias = (0,0.5)
xticks = np.arange(0,q.size+1,int(q.size/4))
yticks = np.arange(0,tau.size+1,int(tau.size/4))
yticks[1:] -= 1
for b in bias:
    p = np.zeros((tau.size,q.size))
    for i,t in enumerate(tau):
        p[i] = calcLogisticProb(q,t,b)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(p,clim=(0,1),cmap='magma',origin='lower',aspect='auto')
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(q[xticks],1))
    ax.set_yticks(yticks)
    ax.set_yticklabels(tau[yticks])
    ax.set_xlabel('Q')
    ax.set_ylabel('temperature')
    ax.set_title('lick probability, bias='+str(b))
    plt.colorbar(im)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for t,clr in zip((0.1,0.2,0.4),'rgb'):
    for b,ls in zip(bias,('-','--')):
        ax.plot(q,calcLogisticProb(q,t,b),color=clr,ls=ls,label='temperature='+str(t)+', bias='+str(b))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(np.arange(-1,1.1,0.5))
ax.set_yticks(np.arange(0,1.1,0.5))
ax.set_xlim([-1,1])
ax.set_ylim([0,1])
ax.set_xlabel('Q',fontsize=14)
ax.set_ylabel('lick probability',fontsize=14)
ax.legend()
plt.tight_layout()


# get fit params from HPC output
trainingPhases = ('initial training','after learning','nogo','noAR','rewardOnly','no reward')
fixedParamNames = ('Full model','visConf','audConf','alphaContext','alphaAction','alphaContext,\nalphaAction','alphaHabit')
fixedParamValues = (None,1,1,0,0,0,0)
nModelParams = (7,6,6,6,6,5,6)
paramNames = ('tauAction','biasAction','visConf','audConf','alphaContext','alphaAction','alphaHabit')
paramBounds = ([0,1],[-1,1],[0.5,1],[0.5,1],[0,1],[0,1],[0,1])
modelData = {phase: {} for phase in trainingPhases}
filePaths = glob.glob(os.path.join(r"\\allen\programs\mindscope\workgroups\dynamicrouting\Sam\RLmodel",'*.npz'))
for f in filePaths:
    mouseId,sessionDate,sessionTime,trainingPhase = os.path.splitext(os.path.basename(f))[0].split('_')
    session = sessionDate+'_'+sessionTime
    with np.load(f) as data:
        params = data['params']
        logLoss = data['logLoss']
    d = modelData[trainingPhase]
    if mouseId not in d:
        d[mouseId] = {session: {'params': params, 'logLoss': logLoss}}
    elif session not in d[mouseId]:
        d[mouseId][session] = {'params': params, 'logLoss': logLoss}
    elif logLoss < d[mouseId][session]['logLoss']:
        d[mouseId][session]['params'] = params
        d[mouseId][session]['logLoss'] = logLoss


# get experiment data and model latent variables
sessionData = {phase: {} for phase in trainingPhases}        
for trainingPhase in trainingPhases:
    print(trainingPhase)
    d = modelData[trainingPhase]
    if len(d) > 0:
        for mouse in d:
            for session in d[mouse]:
                obj = getSessionData(mouse,session)
                s = d[mouse][session]
                s['pContext'] = []
                s['qAction'] = []
                s['expectedValue'] = []
                s['prediction'] = []
                s['wHabit'] = []
                for params in s['params']:
                    pContext,qAction,expectedValue,wHabit,pAction,action = runModel(obj,*params)
                    s['pContext'].append(pContext[0])
                    s['qAction'].append(qAction[0])
                    s['expectedValue'].append(expectedValue[0])
                    s['prediction'].append(pAction[0])
                    s['wHabit'].append(wHabit[0])
                if mouse not in sessionData[trainingPhase]:
                    sessionData[trainingPhase][mouse] = {session: obj}
                elif session not in sessionData[trainingPhase][mouse]:
                    sessionData[trainingPhase][mouse][session] = obj
                            

# plot logloss
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)
xticks = np.arange(len(fixedParamNames))
xlim = [-0.25,xticks[-1]+0.25]
ax.plot(xlim,[0,0],'--',color='0.5')
for trainingPhase,clr in zip(trainingPhases,'mgrgbc'):
    d = modelData[trainingPhase]
    if len(d) > 0:
        val = np.array([np.mean([session['logLoss'] for session in mouse.values()],axis=0) for mouse in d.values()])
        val -= val[:,fixedParamNames.index('Full model')][:,None]
        mean = val.mean(axis=0)
        sem = val.std(axis=0)/(len(val)**0.5)
        ax.plot(xticks,mean,'o',mec=clr,mfc='none',ms=10,mew=2,label=trainingPhase)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(xlim)
ax.set_xticks(xticks)
ax.set_xticklabels([fixedParamNames[0]]+[name+'='+str(val) for name,val in zip(fixedParamNames[1:],fixedParamValues[1:])])
ax.set_ylabel(r'$\Delta$ NLL')
ax.legend(loc='upper left')
plt.tight_layout()

fig = plt.figure(figsize=(5,10))
for i,(fixedParam,fixedVal) in enumerate(zip(fixedParamNames,fixedParamValues)):
    ax = fig.add_subplot(len(fixedParamNames),1,i+1)
    ax.plot([0,0],[0,1],'--',color='0.5')
    for trainingPhase,clr in zip(trainingPhases,'mgrgbc'):
        d = modelData[trainingPhase]
        if len(d) > 0:
            logLoss = np.array([np.mean([session['logLoss'] for session in mouse.values()],axis=0) for mouse in d.values()])          
            logLoss = logLoss[:,i] - logLoss[:,fixedParamNames.index('Full model')] if fixedParam != 'Full model' else logLoss[:,i]
            dsort = np.sort(logLoss)
            cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
            ax.plot(dsort,cumProb,color=clr,label=trainingPhase)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(([0,1.2] if fixedParam == 'Full model' else [-0.05,0.17]))
    ax.set_ylim([0,1.01])
    ax.set_xlabel(('NLL' if fixedParam == 'Full model' else r'$\Delta$ NLL'))
    ax.set_title((fixedParam if fixedParam == 'Full model' else fixedParam+'='+str(fixedVal)))
    if i==0:
        ax.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
                
                
# plot fit param values
fig = plt.figure(figsize=(12,10))
gs = matplotlib.gridspec.GridSpec(len(fixedParamNames),len(paramNames))
for i,(fixedParam,fixedVal) in enumerate(zip(fixedParamNames,fixedParamValues)):
    for j,(param,xlim) in enumerate(zip(paramNames,paramBounds)):
        ax = fig.add_subplot(gs[i,j])
        for trainingPhase,clr in zip(trainingPhases,'mgrgbc'):
            d = modelData[trainingPhase]
            if len(d) > 0:
                paramVals = np.array([np.mean([session['params'][i,j] for session in mouse.values()]) for mouse in d.values()])            
                dsort = np.sort(paramVals)
                cumProb = np.array([np.sum(dsort<=s)/dsort.size for s in dsort])
                ax.plot(dsort,cumProb,color=clr,label=trainingPhase)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([xlim[0]-0.02,xlim[1]+0.02])
        ax.set_ylim([0,1.01])
        if i==len(fixedParamNames)-1:
            ax.set_xlabel(param)
        if j==0:
            ax.set_ylabel('Cum. Prob.')
        if j==3:
            ax.set_title((fixedParam if fixedParam == 'Full model' else fixedParam+'='+str(fixedVal)))
        if i==0 and j==len(paramNames)-1:
            ax.legend(bbox_to_anchor=(1,1))
plt.tight_layout()


# compare model and mice
stimNames = ('vis1','vis2','sound1','sound2')
preTrials = 5
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
                                resp = d[mouse][session][var][fixedParamNames.index(fixedParam)]
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
                            v = d[mouse][session][var][fixedParamNames.index(fixedParam)]
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
        
        
# plot q values
preTrials = 20
postTrials = 60
for trainingPhase in trainingPhases:
    d = modelData[trainingPhase]
    if len(d) == 0:
        continue
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
                    q = d[mouse][session]['qAction'][fixedParamNames.index(fixedParam)].reshape(obj.nTrials,8)
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if rewStim==rewardStim and blockInd > 0:
                            y[-1].append(np.full((8,preTrials+postTrials),np.nan))
                            pre = q[(obj.trialBlock==blockInd)]
                            k = min(preTrials,len(pre))
                            y[-1][-1][:,preTrials-k:preTrials] = pre[-k:].T
                            post = q[(obj.trialBlock==blockInd+1)]
                            k = min(postTrials,len(post))
                            y[-1][-1][:,preTrials:preTrials+k] = post[:k].T
                y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            im = ax.imshow(m,clim=(-1,1),cmap='bwr')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks(np.arange(-20,60,20))
            # ax.set_yticks([0,0.5,1])
            ax.set_xlim([-preTrials-0.5,postTrials+0.5])
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
                cb.set_ticks([-1,0,1])
    plt.tight_layout()


# no reward blocks, target stimuli only
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
                        resp = d[mouse][session]['prediction'][fixedParamNames.index(fixedParam)]
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



def runModelOld(obj,tauAction,biasAction,visConfidence,audConfidence,alphaContext,alphaAction,alphaHabit,useHistory=True,nReps=1):
    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    
    pContext = np.zeros((nReps,obj.nTrials,2)) + 0.5
    
    qAction = -np.ones((nReps,obj.nTrials,2,len(stimNames)),dtype=float)  
    if alphaContext > 0:
        qAction[:,:,0,0] = 1
        qAction[:,:,1,2] = 1
    else:
        qAction[:,:,:,[0,2]] = 1

    expectedValue = -np.ones((nReps,obj.nTrials))

    qHabit = np.array([1,-1,1,-1])
    pHabit = np.zeros((nReps,obj.nTrials))

    pAction = np.zeros((nReps,obj.nTrials))
    
    action = np.zeros((nReps,obj.nTrials),dtype=int)
    
    for i in range(nReps):
        for trial,stim in enumerate(obj.trialStim):
            if stim != 'catch':
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]
                    
                if alphaContext > 0:
                    expectedValue[i,trial] = np.sum(qAction[i,trial] * pStim[None,:] * pContext[i,trial][:,None])
                else:
                    context = 0
                    expectedValue[i,trial] = np.sum(qAction[i,trial,context] * pStim)

                q = (pHabit[i,trial] * np.sum(qHabit * pStim)) + ((1 - pHabit[i,trial]) * expectedValue[i,trial])              
            
                pAction[i,trial] = calcLogisticProb(q,tauAction,biasAction)
                
                if useHistory:
                    action[i,trial] = obj.trialResponse[trial]
                elif random.random() < pAction[i,trial]:
                    action[i,trial] = 1 
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qAction[i,trial+1] = qAction[i,trial]
                pHabit[i,trial+1] = pHabit[i,trial]
            
                if action[i,trial] or obj.autoRewarded[trial]:
                    outcome = 1 if obj.trialRewarded[trial] else -1
                    predictionError = outcome - expectedValue[i,trial]
                    
                    if alphaContext > 0 and stim != 'catch':
                        if outcome < 1:
                            pContext[i,trial+1,modality] -= alphaContext * pStim[0 if modality==0 else 2] * pContext[i,trial,modality]
                        else:
                            pContext[i,trial+1,modality] += alphaContext * (1 - pContext[i,trial,modality]) 
                        pContext[i,trial+1,1 if modality==0 else 0] = 1 - pContext[i,trial+1,modality]
                    
                    if alphaAction > 0 and stim != 'catch':
                        if alphaContext > 0:
                            qAction[i,trial+1] += alphaAction * pStim[None,:] * pContext[i,trial][:,None] * predictionError
                        else:
                            qAction[i,trial+1,context] += alphaAction * pStim * predictionError
                        qAction[i,trial+1][qAction[i,trial+1] > 1] = 1 
                        qAction[i,trial+1][qAction[i,trial+1] < -1] = -1

                    if alphaHabit > 0:
                        pHabit[i,trial+1] += alphaHabit * (0.5 * abs(predictionError) - pHabit[i,trial])
    
    return pContext, qAction, expectedValue, pHabit, pAction, action

