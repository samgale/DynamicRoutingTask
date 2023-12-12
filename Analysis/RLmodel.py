#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:47:48 2023

@author: samgale
"""

import copy
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn.metrics



def softmax(q,tau):
    p = np.exp(q / tau)
    p /= p.sum()
    return p


def softmaxWithBias(q,tau,bias,norm=True):
    p = np.exp((q + bias) / tau)
    p /= p + 1
    if norm:
        low = softmaxWithBias(-1,tau,bias,norm=False)
        high = softmaxWithBias(1,tau,bias,norm=False)
        offset = softmaxWithBias(-1,tau,0,norm=False)
        p -= low
        p *= (1 - 2*offset) / (high - low)
        p += offset
    return p


def fitModel(exps,contextMode,fitParamRanges):
    actualResponse = np.concatenate([obj.trialResponse for obj in exps])
    bestAccuracy = 0
    for params in itertools.product(*fitParamRanges):
        modelResponse = np.concatenate(runModel(exps,contextMode,*params)[0])
        modelAccuracy = sklearn.metrics.accuracy_score(actualResponse,modelResponse)
        if modelAccuracy > bestAccuracy:
            bestAccuracy = modelAccuracy
            bestParams = params
    return bestParams


def runModel(exps,contextMode,visConfidence,audConfidence,alphaContext,tauAction,biasAction,alphaAction,penalty):
    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    
    response = []
    pc = []
    qa = []
    
    for obj in exps:
        response.append(np.zeros(obj.nTrials,dtype=int))
        
        pContext = np.zeros((obj.nTrials,2),dtype=float) + 0.5
        
        Qaction = np.zeros((obj.nTrials,2,len(stimNames)),dtype=float) + penalty
        if contextMode == 'no context':
            Qaction[:,:,[0,2]] = 1
        else:
            Qaction[:,0,0] = 1
            Qaction[:,1,2] = 1
        
        for trial,(stim,rewStim,autoRew) in enumerate(zip(obj.trialStim,obj.rewardedStim,obj.autoRewardScheduled)):
            if stim == 'catch':
                action = 0
            else:
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]
                
                if contextMode == 'switch context':
                    if trial == 0:
                        context = modality
                    else:
                        context = 0 if pContext[trial,0] > 0.5 else 1
                else:
                    context = 0
                    
                if contextMode == 'weight context':
                    q = np.sum(Qaction[trial] * pStim[None,:] * pContext[trial][:,None])
                else:
                    q = np.sum(Qaction[trial,context] * pStim)
                pAction = softmaxWithBias(q,tauAction,biasAction)
                
                action = 1 if autoRew else np.random.choice(2,p=[1-pAction,pAction])
            
            if trial+1 < obj.nTrials:
                pContext[trial+1] = pContext[trial]
                Qaction[trial+1] = Qaction[trial]
            
                if action:
                    outcome = 1 if stim==rewStim else penalty
                    err = outcome - q
                    
                    if contextMode != 'no context':
                        if outcome < 1:
                            pContext[trial+1,modality] -= alphaContext * pStim[0 if modality==0 else 2] * pContext[trial,modality]
                        else:
                            pContext[trial+1,modality] += alphaContext * (1 - pContext[trial,modality]) 
                        pContext[trial+1,1 if modality==0 else 0] = 1 - pContext[trial+1,modality]
                    
                    if contextMode == 'weight context':
                        Qaction[trial+1] += alphaAction * pStim[None,:] * pContext[trial][:,None] * err
                    else:
                        Qaction[trial+1,context] += alphaAction * pStim * err
                    Qaction[trial+1][Qaction[trial+1] >1] = 1 
                    Qaction[trial+1][Qaction[trial+1] <-1] = -1 
            
            response[-1][trial] = action
            
        pc.append(pContext)
        qa.append(Qaction)
    
    return response, pc, qa


# fit model
stages = ('early','late')
contextModes = ('no context','weight context') # ('no context','switch context','weight context')
qModes = ('q update','no q update')
modelParams = {stage: {context: {q: [] for q in qModes} for context in contextModes} for stage in stages}
modelResponse = copy.deepcopy(modelParams)
for i,stage in enumerate(stages):
    for j,contextMode in enumerate(contextModes):
        for k,qMode in enumerate(qModes):
            if (stage=='early' and contextMode!='no context') or (contextMode=='no context' and qMode=='no q update'):
                continue
            visConfidenceRange = np.arange(0.75,1,0.1)
            audConfidenceRange = np.arange(0.65,1,0.1)
            if contextMode == 'no context':
                alphaContextRange = (0,)
            else:
                alphaContextRange = np.arange(0.05,1,0.15) 
            tauActionRange = (0.25,)
            biasActionRange = np.arange(0.05,1,0.15)
            if qMode == 'no q update':
                alphaActionRange = (0,)
            else:
                alphaActionRange = np.arange(0.02,0.11,0.02) if stage=='early' else np.arange(0.05,1,0.15)
            penaltyRange = (-1,)
            fitParamRanges = (visConfidenceRange,audConfidenceRange,
                              alphaContextRange,
                              tauActionRange,biasActionRange,alphaActionRange,
                              penaltyRange)
            for m,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
                exps = exps[:5] if stage=='early' else exps[s:s+5]
                modelParams[stage][contextMode][qMode].append([])
                modelResponse[stage][contextMode][qMode].append([])
                for n,testExp in enumerate(exps):
                    print(i,j,k,m,n)
                    trainExps = [obj for obj in exps if obj is not testExp]
                    fitParams = fitModel(trainExps,contextMode,fitParamRanges)
                    modelParams[stage][contextMode][qMode][-1].append(fitParams)
                    modelResponse[stage][contextMode][qMode][-1].append(np.mean([runModel([testExp],contextMode,*fitParams)[0][0] for _ in range(5)],axis=0))


# plot params
paramNames = ('visConf','audConf','alphaContext','tauAction','biasAction','alphaAction','penalty')
x = np.arange(len(paramNames))
for stage in stages:
    for contextMode in modelParams[stage]:
        for qMode in modelParams[stage][contextMode]:
            if len(modelParams[stage][contextMode][qMode]) > 0:
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                params = np.mean(modelParams[stage][contextMode][qMode],axis=1)
                for p in params:
                    ax.plot(x,p,'o',mec='k',mfc='none')
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xticks(x)
                ax.set_xticklabels(paramNames)
                ax.set_title(stage+', '+contextMode+', '+qMode)
                plt.tight_layout()
                

# compare model and mice
stimNames = ('vis1','vis2','sound1','sound2')

preTrials = 5
postTrials = 15
x = np.arange(-preTrials,postTrials+1)
for stage in stages:
    fig = plt.figure(figsize=(8,8))
    a = 0
    for contextMode in ('mice',) + tuple(modelResponse[stage].keys()):
        for qMode in ((None,) if contextMode=='mice' else modelResponse[stage][contextMode].keys()):
            if contextMode!='mice' and len(modelResponse[stage][contextMode][qMode])==0:
                continue
            for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
                ax = fig.add_subplot(6,2,a+1)
                a += 1
                for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                    y = []
                    for m,(exps,s) in enumerate(zip(sessionData,sessionsToPass)):
                        exps = exps[:5] if stage=='early' else exps[s:s+5]
                        y.append([])
                        for n,obj in enumerate(exps):
                            if contextMode == 'mice':
                                resp = obj.trialResponse
                            else:
                                resp = np.array(modelResponse[stage][contextMode][qMode][m][n])
                            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                                if rewStim==rewardStim and blockInd > 0:
                                    trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                    pre = resp[(obj.trialBlock==blockInd) & trials]
                                    i = min(preTrials,pre.size)
                                    y[-1][-1][:i] = pre[-i:]
                                    post = resp[(obj.trialBlock==blockInd+1) & trials]
                                    i = min(postTrials,post.size)
                                    y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
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
                ax.set_xlabel('Trials after block switch')
                ax.set_ylabel('Response rate')
                if a==1:
                    ax.legend(loc='upper right')
                title = 'mice, '+blockLabel+' (n='+str(len(y))+')' if contextMode=='mice' else contextMode+', '+qMode
                ax.set_title(title)
    plt.tight_layout()


respRate = {contextMode: {lbl: [] for lbl in ('rewarded target stim','unrewarded target stim')} for contextMode in ('mice',)+contextModes}
preTrials = 5
postTrials = 15
x = np.arange(-preTrials,postTrials+1)    
for stage in stages: 
    fig = plt.figure(figsize=(8,8))
    a = 0
    for contextMode in ('mice',) + contextModes:
        if stage=='early' and contextMode=='weight':
            continue
        ax = fig.add_subplot(3,1,a+1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for lbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
            y = []
            for i,exps in enumerate(expsByMouse):
                if len(exps)>0:
                    # exps = exps[:5] if stage=='early' else exps[passSession[i]:passSession[i]+5]
                    y.append([])
                    for j,obj in enumerate(exps):
                        if contextMode == 'mice':
                            resp = obj.trialResponse
                        else:
                            resp = np.array(modelResponse[stage][contextMode][i][j])
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0:
                                stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in lbl else rewStim
                                trials = (obj.trialStim==stim) & ~obj.autoRewardScheduled
                                y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                                pre = resp[(obj.trialBlock==blockInd) & trials]
                                k = min(preTrials,pre.size)
                                y[-1][-1][:k] = pre[-k:]
                                post = resp[(obj.trialBlock==blockInd+1) & trials]
                                k = min(postTrials,post.size)
                                y[-1][-1][preTrials+1:preTrials+1+k] = post[:k]
                    y[-1] = np.nanmean(y[-1],axis=0)
            respRate[contextMode][lbl] = np.array(y)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,label=lbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks(np.arange(-preTrials,postTrials+1,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        if a==2:
            ax.set_xlabel('Trials of indicated type after block switch (auto-rewards excluded)',fontsize=12)
        ax.set_ylabel('Response rate',fontsize=12)
        if contextMode=='mice':
            title = str(len(y))+' mice'
        elif contextMode=='none':
            title = 'Q learning model'
        else:
            title = 'Q learning with context belief model'
        ax.set_title(title,fontsize=12)
        if a==0:
            ax.legend(bbox_to_anchor=(1,1))
        a += 1
    plt.tight_layout()
    

fig = plt.figure(figsize=(6,8))
for i,contextMode in enumerate(respRate.keys()):
    ax = fig.add_subplot(3,1,i+1)
    ax.plot([0.5,0.5],[0,1],'k--')
    for lbl,clr in zip(respRate[contextMode].keys(),'gm'):
        rr = respRate[contextMode][lbl][:,[preTrials-1,preTrials+1]]
        for r in rr:
            ax.plot([0,1],r,'o-',color=clr,mec=clr,mfc='none',ms=5,alpha=0.1)
        mean = np.mean(rr,axis=0)
        sem = np.std(rr,axis=0)/(len(rr)**0.5)
        ax.plot([0,1],mean,'o-',color=clr,ms=10,label=lbl)
        for x,m,s in zip([0,1],mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['last trial before\nblock switch','first trial after\nblock switch'])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-0.25,1.25])
    ax.set_ylim([0,1.01])
    ax.set_ylabel('Response rate',fontsize=12)
    if contextMode=='mice':
        title = str(len(y))+' mice'
    elif contextMode=='none':
        title = 'Q learning model'
    else:
        title = 'Q learning with context belief model'
    ax.set_title(title,fontsize=12)
    if a==0:
        ax.legend(bbox_to_anchor=(1,1))
plt.tight_layout()


    
# plot Q values
Qcontext = {stage: {context: [] for context in contextModes} for stage in stages}
Qaction = copy.deepcopy(Qcontext)
Qweight = copy.deepcopy(Qcontext)
pVis = copy.deepcopy(Qcontext)
pLick = copy.deepcopy(Qcontext)
for s,stage in enumerate(stages):
    for i,contextMode in enumerate(contextModes):
        if stage=='early' and contextMode=='weight':
            continue
        for j,exps in enumerate(expsByMouse):
            exps = exps[:5] if stage=='early' else exps[passSession[j]:passSession[j]+5]
            Qcontext[stage][contextMode].append([])
            Qaction[stage][contextMode].append([])
            Qweight[stage][contextMode].append([])
            pVis[stage][contextMode].append([])
            pLick[stage][contextMode].append([])
            for k,testExp in enumerate(exps):
                print(s,i,j,k)
                fitParams = modelParams[stage][contextMode][j][k]
                qc = []
                qa = []
                qw = []
                pv = []
                pl = []
                for _ in range(5):
                    c,a = runModel([testExp],contextMode,*fitParams)[1:]
                    qa.append(a[0])
                    if contextMode !='none':
                        qc.append(c[0])
                        pc = np.array([softmax(q,fitParams[0]) for q in c[0]])
                        qw.append(np.sum(a[0][:,:,[0,2],1] * pc[:,:,None],axis=1))
                        pv.append(pc[:,0])
                        pl.append(np.array([[softmaxWithBias(q,*fitParams[2:4]) for q in qq] for qq in qw[-1]])) 
                Qcontext[stage][contextMode][-1].append(np.mean(qc,axis=0))
                Qaction[stage][contextMode][-1].append(np.mean(qa,axis=0))
                Qweight[stage][contextMode][-1].append(np.mean(qw,axis=0))
                pVis[stage][contextMode][-1].append(np.mean(pv,axis=0))
                pLick[stage][contextMode][-1].append(np.mean(pl,axis=0))

preTrials = 20
postTrials = 70
x = np.arange(-preTrials,postTrials)    
for stage in stages:
    fig = plt.figure(figsize=(8,6))
    a = 0
    for contextMode in contextModes:
        if stage=='early' and contextMode=='weight':
            continue
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual','auditory')):
            ax = fig.add_subplot(2,2,a+1)
            a += 1
            ax.plot([0,0],[-1,1],':',color='0.7')
            ax.plot([-preTrials-0.5,postTrials+0.5],[0,0],':',color='0.7')
            lines = (('Qv','Qa'),'gm',('-','-')) if contextMode=='none' else (('Q vis context','Qwv','Qwa','Qvv','Qva','Qav','Qaa'),'kbrgmgm',('-','-','-','-','--','--','-'))
            for lbl,clr,ls in zip(*lines):
                y = []
                for i,exps in enumerate(expsByMouse):
                    exps = exps[:5] if stage=='early' else exps[passSession[i]:passSession[i]+5]
                    for j,obj in enumerate(exps):
                        if lbl=='Q vis context':
                            d = Qcontext[stage][contextMode][i][j][:,0]
                        elif lbl=='Qwv':
                            d = Qweight[stage][contextMode][i][j][:,0]
                        elif lbl=='Qwa':
                            d = Qweight[stage][contextMode][i][j][:,1]
                        else:
                            d = Qaction[stage][contextMode][i][j]
                            if lbl in ('Qv','Qvv'):
                                d = d[:,0,0,1]
                            elif lbl in ('Qa','Qva'):
                                d = d[:,0,2,1]
                            elif lbl=='Qav':
                                d = d[:,1,0,1]
                            elif lbl=='Qaa':
                                d = d[:,1,2,1]
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                y.append(np.full(preTrials+postTrials,np.nan))
                                pre = d[obj.trialBlock==blockInd]
                                k = min(preTrials,pre.size)
                                y[-1][:k] = pre[-k:]
                                post = d[obj.trialBlock==blockInd+1]
                                k = min(postTrials,post.size)
                                y[-1][preTrials:preTrials+k] = post[:k]
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x,m,color=clr,ls=ls,label=lbl)
                ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=12)
            ax.set_xlim([-preTrials-0.5,postTrials+0.5])
            ax.set_ylim([-1.01,1.01])
            if (stage=='early' and contextMode=='none') or contextMode=='weight':
                ax.set_xlabel('Trials from block switch',fontsize=12)
            if blockLabel=='visual':
                ax.set_ylabel('Q',fontsize=12)
            if contextMode=='none':
                title = blockLabel+' rewarded blocks\n'+'Q learning'
            else:
                title = 'Q learning with context belief'
            ax.set_title(title,fontsize=12)
            if blockLabel=='auditory':
                ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()


# plot relationship bewtween tau and q values
Q = np.arange(-1,1.01,0.01)

epsilon = (0.1,0.33)
for epsi in epsilon:
    p = np.zeros((Q.size,Q.size))
    for i,qi in enumerate(Q):
        for j,qj in enumerate(Q):
            if qi == qj:
                p[i,j] = 0.5
            else:
                p[i,j] = 1-epsi/2 if qi > qj else epsi/2
            
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(p,clim=(0,1),cmap='magma',origin='lower',aspect='auto')
    ax.set_xticks(np.arange(0,Q.size+1,int(Q.size/4)))
    ax.set_xticklabels(np.arange(-1,1.1,0.5))
    ax.set_yticks(np.arange(0,Q.size+1,int(Q.size/4)))
    ax.set_yticklabels(np.arange(-1,1.1,0.5))
    ax.set_xlabel('Q aud')
    ax.set_ylabel('Q vis')
    ax.set_title('vis probability, epsilon='+str(epsi))
    plt.colorbar(im)

tau = (0.25,1)
for t in tau:
    p = np.zeros((Q.size,Q.size))
    for i,qi in enumerate(Q):
        for j,qj in enumerate(Q):
            p[i,j] = softmax(np.array([qi,qj]),t)[0]
            
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(p,clim=(0,1),cmap='magma',origin='lower',aspect='auto')
    ax.set_xticks(np.arange(0,Q.size+1,int(Q.size/4)))
    ax.set_xticklabels(np.arange(-1,1.1,0.5))
    ax.set_yticks(np.arange(0,Q.size+1,int(Q.size/4)))
    ax.set_yticklabels(np.arange(-1,1.1,0.5))
    ax.set_xlabel('Q aud')
    ax.set_ylabel('Q vis')
    ax.set_title('vis probability, temperature='+str(t))
    plt.colorbar(im)


tau = np.arange(0.01,4.01,0.01)
bias = (0,0.5)
for b in bias:
    p = np.zeros((Q.size,tau.size))
    for i,q in enumerate(Q):
        for j,t in enumerate(tau):
            p[i,j] = softmaxWithBias(q,t,b)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(p,clim=(0,1),cmap='magma',origin='lower',aspect='auto')
    ax.set_xticks(np.arange(0,tau.size+1,int(tau.size/4)))
    ax.set_xticklabels(np.arange(5))
    ax.set_yticks(np.arange(0,Q.size+1,int(Q.size/4)))
    ax.set_yticklabels(np.arange(-1,1.1,0.5))
    ax.set_xlabel('temperature')
    ax.set_ylabel('Q')
    ax.set_title('lick probability, bias='+str(b))
    plt.colorbar(im)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for t,clr in zip((0.25,0.5),'br'):
    for b,ls in zip(bias,('-','--')):
        ax.plot(Q,[softmaxWithBias(q,t,b) for q in Q],color=clr,ls=ls,label='temperature='+str(t)+', bias='+str(b))
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