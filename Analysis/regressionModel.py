# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:39:53 2023

@author: svc_ccg
"""

import copy
import os
import numpy as np
import pandas as pd
import scipy.cluster
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData
import sklearn
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests


# get data
baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"

excelPath = os.path.join(baseDir,'DynamicRoutingTraining.xlsx')
sheets = pd.read_excel(excelPath,sheet_name=None)
allMiceDf = sheets['all mice']

mouseIds = ('638573','638574','638575','638576','638577','638578',
            '649943','653481','656726','658096','659250')

passOnly = True

mice = []
sessionStartTimes = []
passSession =[]
for mid in mouseIds:
    if str(mid) in sheets:
        mouseInd = np.where(allMiceDf['mouse id']==int(mid))[0][0]
        df = sheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        if any('stage 3' in task for task in df['task version']) and not any('stage 4' in task for task in df['task version']):
            sessions[np.where(sessions)[0][0]] = False # skipping first 6-block session when preceded by distractor training
        firstExperimentSession = np.where(['multimodal' in task
                                           or 'contrast'in task
                                           or 'opto' in task
                                           or 'nogo' in task
                                           or 'noAR' in task
                                           #or 'NP' in rig 
                                           for task,rig in zip(df['task version'],df['rig name'])])[0]
        if len(firstExperimentSession)>0:
            sessions[firstExperimentSession[0]:] = False
        if sessions.sum() > 0 and df['pass'][sessions].sum() > 0:
            mice.append(str(mid))
            if passOnly:
                sessions[:np.where(sessions & df['pass'])[0][0]-1] = False
                passSession.append(0)
            else:
                passSession.append(np.where(df['pass'][sessions])[0][0]-1)
            sessionStartTimes.append(list(df['start time'][sessions]))
        
expsByMouse = []
for mid,st in zip(mice,sessionStartTimes):
    expsByMouse.append([])
    for t in st:
        f = os.path.join(baseDir,'Data',mid,'DynamicRouting1_' + mid + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5')
        obj = DynRoutData()
        obj.loadBehavData(f)
        expsByMouse[-1].append(obj)
        
nMice = len(expsByMouse)
nExps = [len(exps) for exps in expsByMouse]


# construct regressors
nTrialsPrev = 15
stimNames = ('vis1','vis2','sound1','sound2')
regressors = ('reinforcement','noReinforcement',
              'crossModalReinforcement','crossModalNoReinforcement',
              'preservation','reward','action')
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
        s += 1
        for blockInd in range(6):
            b += 1
            print(m,s,b)
            if blockInd==0:
                continue
            trials = ~obj.catchTrials & ~obj.autoRewarded & (obj.trialBlock==blockInd+1)
            trialInd = np.where(trials)[0]
            nTrials = trials.sum()
            regData['X'].append({})
            for r in regressors:
                regData['X'][-1][r] = np.zeros((nTrials,nTrialsPrev))
                for n in range(1,nTrialsPrev+1):
                    for trial,stim in enumerate(obj.trialStim[trials]):
                        if r in ('reinforcement','noReinforcement','preservation'):
                            sameStim = obj.trialStim[:trialInd[trial]] == stim
                            if sameStim.sum()>n:
                                resp = obj.trialResponse[:trialInd[trial]][sameStim][-n]
                                rew = obj.trialRewarded[:trialInd[trial]][sameStim][-n]
                                if (r=='reinforcement' and rew) or (r=='noReinforcement' and resp and not rew):
                                    regData['X'][-1][r][trial,n-1] = 1
                                elif r=='preservation' and resp:
                                    regData['X'][-1][r][trial,n-1] = 1
                        elif r in ('crossModalReinforcement','crossModalNoReinforcement'):
                            otherModalTarget = 'vis1' if stim[:-1]=='sound' else 'sound1'
                            otherModal = obj.trialStim[:trialInd[trial]] == otherModalTarget
                            if otherModal.sum()>n:
                                resp = obj.trialResponse[:trialInd[trial]][otherModal][-n]
                                rew = obj.trialRewarded[:trialInd[trial]][otherModal][-n]
                                if (r=='crossModalReinforcement' and rew) or (r=='crossModalNoReinforcement' and resp and not rew):
                                    regData['X'][-1][r][trial,n-1] = 1
                        else:
                            notCatch = obj.trialStim[:trialInd[trial]] != 'catch'
                            if notCatch.sum()>n:
                                resp = obj.trialResponse[:trialInd[trial]][notCatch][-n]
                                rew = obj.trialRewarded[:trialInd[trial]][notCatch][-n]
                                if r=='reward' and rew:
                                    regData['X'][-1][r][trial,n-1] = 1
                                elif r=='action' and resp:
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
fitRegressors = ('reinforcement','noReinforcement',
                 'crossModalReinforcement','crossModalNoReinforcement',
                 'preservation','reward','action')
holdOutRegressor = ('none',)# + fitRegressors + (('reinforcement','noReinforcement'),('crossModalReinforcement','crossModalNoReinforcement'))
accuracy = {h: [] for h in holdOutRegressor}
trainAccuracy = copy.deepcopy(accuracy)
balancedAccuracy = copy.deepcopy(accuracy)
prediction = copy.deepcopy(accuracy)
featureWeights = copy.deepcopy(accuracy)
bias = copy.deepcopy(accuracy)
for h in holdOutRegressor:
    # predict each block by fitting all other blocks from the same mouse
    for m in np.unique(regData['mouseIndex']):
        x = []
        y = []
        for b in range(len(regData['blockIndex'])):
            if regData['mouseIndex'][b]==m:
                x.append(np.concatenate([regData['X'][b][r] for r in fitRegressors if r!=h and r not in h],axis=1))
                y.append(regData['trialResponse'][b])
        regMeans = np.mean(np.concatenate(x),axis=0)
        for i in range(len(x)):
            print(h,m,i)
            trainX = np.concatenate(x[:i]+x[i+1:])
            trainX -= regMeans
            trainY = np.concatenate(y[:i]+y[i+1:])
            testX = x[i] - regMeans
            testY = y[i]
            model = LogisticRegression(fit_intercept=True,max_iter=1e3)
            model.fit(trainX,trainY)
            trainAccuracy[h].append(model.score(trainX,trainY))
            accuracy[h].append(model.score(testX,testY))
            prediction[h].append(model.predict(testX))
            balancedAccuracy[h].append(sklearn.metrics.balanced_accuracy_score(testY,prediction[h][-1]))
            featureWeights[h].append(model.coef_[0])
            bias[h].append(model.intercept_)
    

# plots
regressorColors = 'rgmbyck'[:len(fitRegressors)]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for x,h in enumerate(holdOutRegressor):
    a = balancedAccuracy[h]
    m = np.mean(a)
    s = np.std(a)/(len(a)**0.5)
    ax.plot(x,m,'ko')
    ax.plot([x,x],[m-s,m+s],'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.5,1])
ax.set_xticks(np.arange(len(holdOutRegressor)))
ax.set_xticklabels(holdOutRegressor)
ax.set_ylabel('Balanced accuracy')
plt.tight_layout()


x = np.arange(nTrialsPrev)+1
for h in holdOutRegressor:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # m = np.mean(bias[h])
    # s = np.std(bias[h])/(len(bias[h])**0.5)
    # ax.plot([x[0],x[-1]],[m,m],color='0.7')
    # ax.fill_between([x[0],x[-1]],[m+s]*2,[m-s]*2,color='0.7',alpha=0.25)
    d = np.array(featureWeights[h])
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
    ax.legend(title='features')
    ax.set_title(h)
    plt.tight_layout()
    break

#todo: make sum of regressors plot


postTrials = 15
x = np.arange(postTrials)+1
for h in holdOutRegressor:
    fig = plt.figure()
    f = 1
    for d,ylbl in zip((regData['trialResponse'],prediction[h]),('mice','model')):
        for rewStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
            ax = fig.add_subplot(2,2,f)
            for stim,clr,ls,lbl in zip(stimNames,'ggmm',('-','--','-','--'),('visual go','visual nogo','auditory go','auditory nogo')):
                resp = []
                for j,r in enumerate(d):
                    if regData['rewardStim'][j]==rewStim:
                        resp.append(np.full(postTrials,np.nan))
                        a = r[regData['trialStim'][j]==stim][:postTrials]
                        resp[-1][:len(a)] = a
                m = np.nanmean(resp,axis=0)
                s = np.nanstd(resp)/(len(resp)**0.5)
                ax.plot(x,m,color=clr,ls=ls,label=lbl)
                ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([0.5,postTrials+0.5])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Trials after block switch cue trials')
            ax.set_ylabel('Response rate of '+ylbl)
            ax.legend(loc='lower right')
            ax.set_title(str(h)+'\n'+blockLabel+' (n='+str(len(resp))+')')
            plt.tight_layout()
            f+=1
    break





































