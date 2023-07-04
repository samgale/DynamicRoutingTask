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


expsByMouse = [exps for lbl in sessionData for exps in sessionData[lbl]]

expsByMouse = sessionData['noAR']


# construct regressors
nTrialsPrev = 10
reinforcementForgetting = False
regressors = ('reinforcement','crossModalReinforcement',
              'posReinforcement','negReinforcement',
              'crossModalPosReinforcement','crossModalNegReinforcement',
              'preservation','reward','action','stimulus','catch')
regData = {}
regData['mouseIndex'] = []
regData['sessionIndex'] = []
regData['blockIndex'] = []
regData['sessionNumber'] = []
regData['blockNumber'] = []
regData['rewardStim'] = []
regData['trialStim'] = []
regData['trialResponse'] = []
regData['X'] = []
s = -1
b = -1
for m,exps in enumerate(expsByMouse):
    for sn,obj in enumerate(exps):
        print(m,sn)
        for blockInd in range(6):
            b += 1
            if blockInd==0:
                continue
            trials = ~obj.catchTrials & ~obj.autoRewarded & (obj.trialBlock==blockInd+1) & np.in1d(obj.trialStim,obj.blockStimRewarded)
            trialInd = np.where(trials)[0]
            nTrials = trials.sum()
            regData['X'].append({})
            for r in regressors:
                regData['X'][-1][r] = np.zeros((nTrials,nTrialsPrev))
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
                                    regData['X'][-1][r][trial,n-1] = 1 if rew[-n] else -1
                                elif r=='posReinforcement' and sameStim[-n] and rew[-n]:
                                    regData['X'][-1][r][trial,n-1] = 1
                                elif r=='negReinforcement' and sameStim[-n] and resp[-n] and not rew[-n]:
                                    regData['X'][-1][r][trial,n-1] = 1  
                                elif r=='crossModalReinforcement' and otherModal[-n]:
                                    regData['X'][-1][r][trial,n-1] = 1 if rew[-n] else -1
                                elif r=='crossModalPosReinforcement' and otherModal[-n] and rew[-n]:
                                    regData['X'][-1][r][trial,n-1] = 1
                                elif r=='crossModalNegReinforcement' and otherModal[-n] and resp[-n] and not rew[-n]:
                                    regData['X'][-1][r][trial,n-1] = 1
                                elif r=='preservation' and sameStim[-n] and resp[-n]:
                                    regData['X'][-1][r][trial,n-1] = 1
                            else:
                                if r=='reinforcement' and resp[sameStim][-n]:
                                    regData['X'][-1][r][trial,n-1] = 1 if rew[sameStim][-n] else -1
                                elif r=='posReinforcement' and rew[sameStim][-n]:
                                    regData['X'][-1][r][trial,n-1] = 1
                                elif r=='negReinforcement' and resp[sameStim][-n] and not rew[sameStim][-n]:
                                    regData['X'][-1][r][trial,n-1] = 1
                                elif r=='crossModalReinforcement' and resp[otherModal][-n]:
                                    regData['X'][-1][r][trial,n-1] = 1 if rew[otherModal][-n] else -1
                                elif r=='crossModalPosReinforcement' and rew[otherModal][-n]:
                                    regData['X'][-1][r][trial,n-1] = 1
                                elif r=='crossModalNegReinforcement' and resp[otherModal][-n] and not rew[otherModal][-n]:
                                    regData['X'][-1][r][trial,n-1] = 1
                                elif r=='preservation' and resp[sameStim][-n]:
                                    regData['X'][-1][r][trial,n-1] = 1
                        elif r=='reward' and rew[-n]:
                            regData['X'][-1][r][trial,n-1] = 1
                        elif r=='action' and resp[-n]:
                            regData['X'][-1][r][trial,n-1] = 1
                        elif r == 'stimulus' and sameStim[-n]: 
                            regData['X'][-1][r][trial,n-1] = 1
                        elif r == 'catch' and trialStim[-n]=='catch': 
                            regData['X'][-1][r][trial,n-1] = 1
            regData['mouseIndex'].append(m)
            regData['sessionIndex'].append(s)
            regData['blockIndex'].append(b)
            regData['blockNumber'].append(blockInd+1)
            regData['sessionNumber'].append(sn+1)
            regData['rewardStim'].append(obj.blockStimRewarded[blockInd])
            regData['trialStim'].append(obj.trialStim[trials])
            regData['trialResponse'].append(obj.trialResponse[trials])    


# fit model
fitRegressors = ('posReinforcement','negReinforcement',
                 'crossModalPosReinforcement','crossModalNegReinforcement')
holdOutRegressor = ('none',) + fitRegressors
accuracy = {h: [] for h in holdOutRegressor}
trainAccuracy = copy.deepcopy(accuracy)
balancedAccuracy = copy.deepcopy(accuracy)
prediction = copy.deepcopy(accuracy)
featureWeights = copy.deepcopy(accuracy)
bias = copy.deepcopy(accuracy)
for h in holdOutRegressor:
    # predict each block by fitting all other blocks from the same mouse
    for m in np.unique(regData['mouseIndex']):
        print(h,m)
        x = []
        y = []
        for b in range(len(regData['blockIndex'])):
            if regData['mouseIndex'][b]==m:
                x.append(np.concatenate([regData['X'][b][r] for r in fitRegressors if r!=h and r not in h],axis=1))
                y.append(regData['trialResponse'][b])
        regMeans = np.mean(np.concatenate(x),axis=0)
        for i in range(len(x)):
            trainX = np.concatenate(x[:i]+x[i+1:])
            trainX -= regMeans
            trainY = np.concatenate(y[:i]+y[i+1:])
            testX = x[i] - regMeans
            testY = y[i]
            model = LogisticRegression(fit_intercept=True,class_weight='none',max_iter=1e3)
            model.fit(trainX,trainY)
            trainAccuracy[h].append(model.score(trainX,trainY))
            accuracy[h].append(model.score(testX,testY))
            prediction[h].append(model.predict(testX))
            balancedAccuracy[h].append(sklearn.metrics.balanced_accuracy_score(testY,prediction[h][-1]))
            featureWeights[h].append(model.coef_[0])
            bias[h].append(model.intercept_)
    

# plots
regressorColors = ([s for s in 'rgmbyck']+['0.5'])[:len(fitRegressors)]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for x,h in enumerate(holdOutRegressor):
    for ac,mfc in zip((accuracy,trainAccuracy),('k','none')):
        d = [np.mean([a for i,a in enumerate(ac[h]) if regData['mouseIndex'][i]==m]) for m in np.unique(regData['mouseIndex'])]
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax.plot(x,m,'o',mec='k',mfc=mfc)
        ax.plot([x,x],[m-s,m+s],'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.5,1])
ax.set_xticks(np.arange(len(holdOutRegressor)))
ax.set_xticklabels(holdOutRegressor)
ax.set_ylabel('Accuracy')
plt.tight_layout()


x = np.arange(nTrialsPrev)+1
for h in holdOutRegressor:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    d = [np.mean([b for i,b in enumerate(bias[h]) if regData['mouseIndex'][i]==m]) for m in np.unique(regData['mouseIndex'])]
    m = np.mean(d)
    s = np.std(d)/(len(d)**0.5)
    ax.plot([x[0],x[-1]],[m,m],color='0.7')
    ax.fill_between([x[0],x[-1]],[m+s]*2,[m-s]*2,color='0.7',alpha=0.25)
    d = [np.mean([fw for i,fw in enumerate(featureWeights[h]) if regData['mouseIndex'][i]==m],axis=0) for m in np.unique(regData['mouseIndex'])]
    reg,clrs = zip(*[(r,c) for r,c in zip(fitRegressors,regressorColors) if r!=h and r not in h])
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
    break


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
        plt.tight_layout()
    break










