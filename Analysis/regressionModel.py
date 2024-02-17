# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:39:53 2023

@author: svc_ccg
"""

import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn
from sklearn.linear_model import LogisticRegression


trainingPhases = ('initial training','after learning')

# construct regressors
nTrialsPrev = 20
reinforcementForgetting = True
regressors = ('context','reinforcement','crossModalReinforcement',
              'posReinforcement','negReinforcement',
              'crossModalPosReinforcement','crossModalNegReinforcement',
              'preservation','reward','action','stimulus','catch')

regData = {phase: {} for phase in trainingPhases}
for phase in regData:
    regData[phase]['mouseIndex'] = []
    regData[phase]['sessionIndex'] = []
    regData[phase]['blockIndex'] = []
    regData[phase]['sessionNumber'] = []
    regData[phase]['blockNumber'] = []
    regData[phase]['rewardStim'] = []
    regData[phase]['trialStim'] = []
    regData[phase]['trialResponse'] = []
    regData[phase]['X'] = []
    s = -1
    b = -1
    for m,exps in enumerate(sessionData):
        if phase == 'initial training':
            exps = exps[:5]
        elif phase == 'after learning':
            exps = exps[sessionsToPass[m]:]
        for sn,obj in enumerate(exps):
            print(phase,m,sn)
            s += 1
            for blockInd in range(6):
                b += 1
                if blockInd==0:
                    continue
                trials = ~obj.catchTrials & ~obj.autoRewardScheduled & (obj.trialBlock==blockInd+1) & np.in1d(obj.trialStim,obj.blockStimRewarded)
                if not np.any(obj.trialResponse[trials]):
                    continue
                trialInd = np.where(trials)[0]
                nTrials = trials.sum()
                regData[phase]['X'].append({})
                regData[phase]['X'][-1]['context'] = (obj.trialStim[trials] == obj.blockStimRewarded[blockInd]).astype(float)[:,None]
                for r in regressors[1:]:
                    regData[phase]['X'][-1][r] = np.zeros((nTrials,nTrialsPrev))
                    for n in range(1,nTrialsPrev+1):
                        for trial,stim in enumerate(obj.trialStim[trials]):
                            resp = obj.trialResponse[:trialInd[trial]]
                            rew = obj.trialRewarded[:trialInd[trial]]
                            trialStim = obj.trialStim[:trialInd[trial]]
                            sameStim = trialStim==stim
                            otherModalTarget = 'vis1' if stim[:-1]=='sound' else 'sound1'
                            otherModal = trialStim==otherModalTarget
                            if 'inforcement' in r or r=='preservation':
                                if reinforcementForgetting:
                                    if r=='reinforcement' and sameStim[-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1 if rew[-n] else (-1 if resp[-n] else 0)
                                    elif r=='posReinforcement' and sameStim[-n] and rew[-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1
                                    elif r=='negReinforcement' and sameStim[-n] and resp[-n] and not rew[-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1  
                                    elif r=='crossModalReinforcement' and otherModal[-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1 if rew[-n] else (-1 if resp[-n] else 0)
                                    elif r=='crossModalPosReinforcement' and otherModal[-n] and rew[-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1
                                    elif r=='crossModalNegReinforcement' and otherModal[-n] and resp[-n] and not rew[-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1
                                    elif r=='preservation' and sameStim[-n] and resp[-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1
                                else:
                                    if r=='reinforcement':
                                        regData[phase]['X'][-1][r][trial,n-1] = 1 if rew[sameStim][-n] else (-1 if resp[-n] else 0)
                                    elif r=='posReinforcement' and rew[sameStim][-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1
                                    elif r=='negReinforcement' and resp[sameStim][-n] and not rew[sameStim][-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1
                                    elif r=='crossModalReinforcement':
                                        regData[phase]['X'][-1][r][trial,n-1] = 1 if rew[otherModal][-n] else (-1 if resp[-n] else 0)
                                    elif r=='crossModalPosReinforcement' and rew[otherModal][-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1
                                    elif r=='crossModalNegReinforcement' and resp[otherModal][-n] and not rew[otherModal][-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1
                                    elif r=='preservation' and resp[sameStim][-n]:
                                        regData[phase]['X'][-1][r][trial,n-1] = 1
                            elif r=='reward' and rew[-n]:
                                regData[phase]['X'][-1][r][trial,n-1] = 1
                            elif r=='action' and resp[-n]:
                                regData[phase]['X'][-1][r][trial,n-1] = 1
                            elif r == 'stimulus' and sameStim[-n]: 
                                regData[phase]['X'][-1][r][trial,n-1] = 1
                            elif r == 'catch' and trialStim[-n]=='catch': 
                                regData[phase]['X'][-1][r][trial,n-1] = 1
                regData[phase]['mouseIndex'].append(m)
                regData[phase]['sessionIndex'].append(s)
                regData[phase]['blockIndex'].append(b)
                regData[phase]['blockNumber'].append(blockInd+1)
                regData[phase]['sessionNumber'].append(sn+1)
                regData[phase]['rewardStim'].append(obj.blockStimRewarded[blockInd])
                regData[phase]['trialStim'].append(obj.trialStim[trials])
                regData[phase]['trialResponse'].append(obj.trialResponse[trials])    


# fit model
fitRegressors = ('reinforcement','crossModalReinforcement','reward','action')
holdOutRegressor = ('none',) + fitRegressors + (('reinforcement','crossModalReinforcement'),('reward','action'))

accuracy = {phase: {h: [] for h in holdOutRegressor} for phase in trainingPhases}
trainAccuracy = copy.deepcopy(accuracy)
balancedAccuracy = copy.deepcopy(accuracy)
logLoss = copy.deepcopy(accuracy)
featureWeights = copy.deepcopy(accuracy)
bias = copy.deepcopy(accuracy)
prediction = copy.deepcopy(accuracy)
predictionProb = copy.deepcopy(accuracy)

for phase in trainingPhases:
    mi = np.array(regData[phase]['mouseIndex'])
    si = np.array(regData[phase]['sessionIndex'])
    for h in holdOutRegressor:
        # predict blocks from each session by fitting all other blocks from the same mouse
        for m in np.unique(mi):
            print(phase,h,m)
            
            accuracy[phase][h].append([])
            trainAccuracy[phase][h].append([])
            balancedAccuracy[phase][h].append([])
            logLoss[phase][h].append([])
            featureWeights[phase][h].append([])
            bias[phase][h].append([])
            
            x = []
            y = []
            ntrials = []
            for s in np.unique(si[mi==m]):
                x.append([])
                y.append([])
                for b in np.where(si==s)[0]:
                    if len(fitRegressors) - (len(h) if isinstance(h,tuple) else 1) > 1:
                        x[-1].append(np.concatenate([regData[phase]['X'][b][r] for r in fitRegressors if r!=h and r not in h],axis=1))
                    else:
                        x[-1].append(regData[phase]['X'][b][[r for r in fitRegressors if r not in h][0]])
                    y[-1].append(regData[phase]['trialResponse'][b])
                ntrials.append([len(b) for b in x[-1]])
                x[-1] = np.concatenate(x[-1])
                y[-1] = np.concatenate(y[-1])
            for i in range(len(x)):
                trainX = np.concatenate(x[:i]+x[i+1:])
                trainY = np.concatenate(y[:i]+y[i+1:])
                testX = x[i]
                testY = y[i]
                model = LogisticRegression(penalty='l2',fit_intercept=True,class_weight=None,max_iter=1e3)
                model.fit(trainX,trainY)
                trainAccuracy[phase][h][-1].append(model.score(trainX,trainY))
                accuracy[phase][h][-1].append(model.score(testX,testY))
                pred = model.predict(testX)
                balancedAccuracy[phase][h][-1].append(sklearn.metrics.balanced_accuracy_score(testY,pred))
                predProb = model.predict_proba(testX)[:,1]
                # logLoss[phase][h][-1].append(sklearn.metrics.log_loss(testY,predProb))
                featureWeights[phase][h][-1].append(model.coef_[0])
                bias[phase][h][-1].append(model.intercept_)
                nstart = 0
                for n in ntrials[i]:
                    prediction[phase][h].append(pred[nstart:nstart+n])
                    predictionProb[phase][h].append(predProb[nstart:nstart+n])
                    nstart += n
    

# plots
regressorColors = ([s for s in 'rgmbyck']+['0.5'])[:len(fitRegressors)]
for phase in trainingPhases:
    fig = plt.figure(figsize=(5,8))
    ax = fig.add_subplot(1,1,1)
    for x,h in enumerate(holdOutRegressor):
        d = [np.mean(a) for a in accuracy[phase][h]]
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax.plot(x,m,'ko')
        ax.plot([x,x],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(len(holdOutRegressor)))
    ax.set_xticklabels(holdOutRegressor,rotation=45,ha='right')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()


x = np.arange(nTrialsPrev)+1
for h in holdOutRegressor:
    fig = plt.figure()
    for i,phase in enumerate(trainingPhases):
        ax = fig.add_subplot(2,1,i+1)
        d = [np.mean(b) for b in bias[phase][h]]
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax.plot([x[0],x[-1]],[m,m],color='0.7')
        ax.fill_between([x[0],x[-1]],[m+s]*2,[m-s]*2,color='0.7',alpha=0.25)
        if 'context' not in fitRegressors or h == 'context':
            d = [np.mean(fw,axis=0) for fw in featureWeights[phase][h]]
            reg,clrs = zip(*[(r,c) for r,c in zip(fitRegressors,regressorColors) if r!=h and r not in h])
        else:
            d = [np.mean(fw,axis=0)[1:] for fw in featureWeights[phase][h]]
            reg,clrs = zip(*[(r,c) for r,c in zip(fitRegressors[1:],regressorColors[1:]) if r!=h and r not in h])
        mean = np.mean(d,axis=0)
        sem = np.std(d,axis=0)/(len(d)**0.5)
        for m,s,clr,lbl in zip(mean.reshape(len(reg),-1),sem.reshape(len(reg),-1),clrs,reg):
            ax.plot(x,m,color=clr,label=lbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0.5,nTrialsPrev+0.5])
        # ax.set_ylim([-0.15,0.8])
        ax.set_xlabel('Trials previous')
        ax.set_ylabel('Feature weight')
        ax.legend(title='features',loc='upper right')
        ax.set_title(h)
    plt.tight_layout()


postTrials = 15
x = np.arange(postTrials)+1
for h in holdOutRegressor:
    fig = plt.figure()
    for i,(d,ylbl) in enumerate(zip((regData['trialResponse'],prediction[h]),('mice','model'))):
        ax = fig.add_subplot(2,1,i+1)
        for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
            y = []
            for m in np.unique(regData['mouseIndex']):
                resp = []
                for j,r in enumerate(d): #range(len(regData['blockIndex'])):
                    if regData['mouseIndex'][j]==m:
                        rewStim = regData['rewardStim'][j]
                        nonRewStim = np.setdiff1d(('vis1','sound1'),rewStim)
                        stim =  nonRewStim if 'unrewarded' in stimLbl else rewStim
                        resp.append(np.full(postTrials,np.nan))
                        a = r[regData['trialStim'][j]==stim][:postTrials]
                        resp[-1][:len(a)] = a
                y.append(np.nanmean(resp,axis=0))
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y)/(len(y)**0.5)
            ax.plot(x,m,color=clr,label=stimLbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(-20,21,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        if i==1:
            ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
        ax.set_ylabel('Response rate of '+ylbl,fontsize=12)
        if i==0:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    ax.set_title(h)
    plt.tight_layout()
    break










