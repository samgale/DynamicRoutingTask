# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:55:19 2022

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


# get data
baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"

mouseIds = ('638573','638574','638575','638576','638577','638578')

excelPath = os.path.join(baseDir,'DynamicRoutingTraining.xlsx')
sheets = pd.read_excel(excelPath,sheet_name=None)
allMiceDf = sheets['all mice']

mice = []
sessionStartTimes = []
passSession =[]
for mid in mouseIds:
    mouseInd = np.where(allMiceDf['mouse id']==int(mid))[0][0]
    df = sheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']])
    sessions[np.where(sessions)[0][0]] = False # skipping first 6-block session because these mice had distractor training
    firstMultiModal = np.where(['multimodal' in task for task in df['task version']])[0]
    if len(firstMultiModal)>0:
        sessions[firstMultiModal[0]:] = False
    if sessions.sum() > 0:
        mice.append(str(mid))
        sessionStartTimes.append(list(df['start time'][sessions]))
        passSession.append(np.where(df['pass'][sessions])[0][0]-1)
        
expsByMouse = []
for mid,st in zip(mice,sessionStartTimes):
    expsByMouse.append([])
    for t in st:
        f = os.path.join(baseDir,'Data',mid,'DynamicRouting1_' + mid + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5')
        obj = DynRoutData()
        obj.loadBehavData(f)
        expsByMouse[-1].append(obj)


# make regressors
nTrialsPrev = 15
regressors = ('reinforcement','attention','persistence')
regressorColors = ('k',)
for r,cm in zip(regressors,(plt.cm.autumn,plt.cm.winter,plt.cm.summer)):
    regressorColors += tuple(cm(np.linspace(0,1,nTrialsPrev))[:,:3])

nMice = len(expsByMouse)
nExps = [len(exps) for exps in expsByMouse]
trialsPerSession = [[] for _ in range(nMice)]
trialsPerBlock = copy.deepcopy(trialsPerSession)
trialStim = copy.deepcopy(trialsPerSession)
trialRewardStim = copy.deepcopy(trialsPerSession)
X = [{r: [] for r in regressors} for _ in range(nMice)]
Y = copy.deepcopy(trialsPerSession)
for m,exps in enumerate(expsByMouse):
    for obj in exps:
        trials = ~obj.catchTrials & ~obj.autoRewarded & (obj.trialBlock>1)
        trialInd = np.where(trials)[0]
        nTrials = trials.sum()
        for r in regressors:
            X[m][r].append(np.zeros((nTrials,nTrialsPrev)))
            for n in range(1,nTrialsPrev+1):
                for trial,stim in enumerate(obj.trialStim[trials]):
                    if r in ('reinforcement','persistence'):
                        sameStim = obj.trialStim[:trialInd[trial]] == stim
                        if sameStim.sum()>n:
                            if r=='reinforcement':
                                if obj.trialResponse[:trialInd[trial]][sameStim][-n]:
                                    X[m][r][-1][trial,n-1] = 1 if obj.trialRewarded[:trialInd[trial]][sameStim][-n] else -1
                            elif r=='persistence':
                                X[m][r][-1][trial,n-1] = obj.trialResponse[:trialInd[trial]][sameStim][-n]
                    elif r=='attention':
                        notCatch = obj.trialStim[:trialInd[trial]] != 'catch'
                        if notCatch.sum()>n:
                            if obj.trialRewarded[:trialInd[trial]][notCatch][-n]:
                                sameModal = any(s in stim and s in obj.trialStim[:trialInd[trial]][notCatch][-n] for s in ('vis','sound'))
                                X[m][r][-1][trial,n-1] = 1 if sameModal else -1
        Y[m].append(obj.trialResponse[trials].astype(float))
        trialsPerSession[m].append(nTrials)
        trialsPerBlock[m].append([np.sum(obj.trialBlock[trials]==block) for block in np.unique(obj.trialBlock[trials])])
        trialStim[m].append(obj.trialStim[trials])
        trialRewardStim[m].append(obj.rewardedStim[trials])


# fit model
holdOutRegressor = ('none',)+regressors+(('reinforcement','attention'),('reinforcement','persistence'),('attention','persistence'))
holdOutColors = 'krgbymc'
accuracy = {h: [[] for m in range(nMice)] for h in holdOutRegressor}
balancedAccuracy = copy.deepcopy(accuracy)
prediction = copy.deepcopy(accuracy)
predictProb = copy.deepcopy(accuracy)
confidence = copy.deepcopy(accuracy)
featureWeights = copy.deepcopy(accuracy)
for h in holdOutRegressor:
    for m in range(nMice):
        for i in range(nExps[m]):
            firstTrial = 0
            for j,blockTrials in enumerate(trialsPerBlock[m][i]):
                print(h,m,i,j)
                x = np.concatenate([X[m][r][i][firstTrial:firstTrial+blockTrials] for r in regressors if r!=h and r not in h],axis=1)
                y = Y[m][i][firstTrial:firstTrial+blockTrials]
                firstTrial += blockTrials
                model = LogisticRegression(fit_intercept=True,max_iter=1e3)
                model.fit(x,y)
                accuracy[h][m].append(model.score(x,y))
                prediction[h][m].append(model.predict(x))
                balancedAccuracy[h][m].append(sklearn.metrics.balanced_accuracy_score(y,model.predict(x)))
                predictProb[h][m].append(model.predict_proba(x))
                confidence[h][m].append(model.decision_function(x))
                featureWeights[h][m].append(model.coef_.flatten())
                

# plot mouse learning curve
for ylbl in ('d\' same modality','d\' other modality'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((nMice,max(nExps)),np.nan)
    for i,(exps,clr) in enumerate(zip(expsByMouse,plt.cm.tab20(np.linspace(0,1,nMice)))):
        if 'same' in ylbl:
            d = [np.mean(obj.dprimeSameModal) for obj in exps]
        else:
            d = [np.mean(obj.dprimeOtherModalGo) for obj in exps]
        ax.plot(np.arange(len(d))+1,d,color=clr,alpha=0.25)
        ax.plot(passSession[i]+1,d[passSession[i]],'o',ms=10,color=clr,alpha=0.25)
        dp[i,:len(d)] = d
    m = np.nanmean(dp,axis=0)
    ax.plot(np.arange(len(m))+1,m,'k',lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,len(m)+1])
    ax.set_ylim([0,4])
    ax.set_xlabel('Session')
    ax.set_ylabel(ylbl)
    plt.tight_layout()
    

# plot average response rate across blocks for mice and model
holdOut = 'none'
stimNames = ('vis1','vis2','sound1','sound2')
postTrials = 15
x = np.arange(postTrials)+1
for ylbl in ('mice','model'):
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for stim,clr,ls,lbl in zip(stimNames,'ggmm',('-','--','-','--'),('visual go','visual nogo','auditory go','auditory nogo')):
            resp = []
            for m in range(nMice):
                block = 0
                for i in range(nExps[m]):
                    # if i>4:
                    #     block += 5
                    #     continue
                    if i<passSession[m]:
                        block += 5
                        continue
                    firstTrial = 0
                    for blockTrials in trialsPerBlock[m][i]:
                        trials = slice(firstTrial,firstTrial+blockTrials)
                        firstTrial += blockTrials
                        stimTrials = trialStim[m][i][trials]==stim
                        if trialRewardStim[m][i][trials][0]==rewardStim:
                            if ylbl=='mice':
                                r = Y[m][i][trials]
                            else:
                                r = prediction[holdOut][m][block]
                            r = r[stimTrials][:postTrials]
                            resp.append(np.full(postTrials,np.nan))
                            resp[-1][:r.size] = r
                        block += 1
            m = np.nanmean(resp,axis=0)
            s = np.nanstd(resp)/(len(resp)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=lbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials after block switch autorewards')
        ax.set_ylabel('Response rate of '+ylbl)
        ax.legend(loc='lower right')
        ax.set_title(blockLabel+' (n='+str(len(resp))+')')
        plt.tight_layout()


# plot model accuracy for holdout regressor
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for h,clr in zip(holdOutRegressor,holdOutColors):
    for m in range(nMice):
        lbl = h if m==0 else None
        a = np.array(balancedAccuracy[h][m]).reshape(5,-1).mean(axis=0)
        ax.plot(np.arange(a.size)+1,a,color=clr,alpha=0.25,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.5,1])
ax.set_xlabel('Session')
ax.set_ylabel('Balanced accuracy')
ax.legend(title='holdout regressor(s)')
plt.tight_layout()


# plot feature weights
x = np.arange(nTrialsPrev)+1
fwEarly = []
fwLate = []
for m in range(nMice):
    block = 0
    for i in range(nExps[m]):
        for _ in range(5):
            if i<5:
                fwEarly.append(featureWeights['none'][m][block])
            elif i>=passSession[m]:
                fwLate.append(featureWeights['none'][m][block])
            block += 1
for fw,lbl in zip((fwEarly,fwLate),('first 5 sessions','after learning')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    mean = np.mean(fw,axis=0)
    sem = np.std(fw,axis=0)/(len(fw)**0.5)
    for m,s,clr in zip(mean.reshape(3,-1),sem.reshape(3,-1),'rgb'):
        ax.plot(x,m,color=clr)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)

plt.plot(np.mean(fwEarly,axis=0).reshape(3,-1).mean(axis=1))


# cluster fwLate
def cluster(data,nClusters=None,method='ward',metric='euclidean',plot=False,colors=None,nreps=1000,labels=None):
    # data is n samples x m parameters
    linkageMat = scipy.cluster.hierarchy.linkage(data,method=method,metric=metric)
    if nClusters is None:
        clustId = None
    else:
        clustId = scipy.cluster.hierarchy.fcluster(linkageMat,nClusters,'maxclust')
    if plot:
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        colorThresh = 0 if nClusters<2 else linkageMat[::-1,2][nClusters-2]
        if colors is not None:
            scipy.cluster.hierarchy.set_link_color_palette(list(colors))
        if labels=='off':
            labels=None
            noLabels=True
        else:
            noLabels=False
        scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,color_threshold=colorThresh,above_threshold_color='k',labels=labels,no_labels=noLabels)
        scipy.cluster.hierarchy.set_link_color_palette(None)
        ax.set_yticks([])
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
        plt.tight_layout()
        
        if nreps>0:
            randLinkage = np.zeros((nreps,linkageMat.shape[0]))
            shuffledData = data.copy()
            for i in range(nreps):
                for j in range(data.shape[1]):
                    shuffledData[:,j] = data[np.random.permutation(data.shape[0]),j]
                _,m = cluster(shuffledData,method=method,metric=metric)
                randLinkage[i] = m[::-1,2]
            
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            k = np.arange(linkageMat.shape[0])+2
            ax.plot(k,np.percentile(randLinkage,2.5,axis=0),'k--')
            ax.plot(k,np.percentile(randLinkage,97.5,axis=0),'k--')
            ax.plot(k,linkageMat[::-1,2],'ko-',mfc='none',ms=10,mew=2,linewidth=2)
            ax.set_xlim([0,k[-1]+1])
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Linkage Distance')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            plt.tight_layout()
    
    return clustId,linkageMat


clustData = np.array(fwLate)

clustId,linkageMat = cluster(clustData,nClusters=3,plot=False)


for clust in np.unique(clustId):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fw = clustData[clustId==clust]
    mean = np.mean(fw,axis=0)
    sem = np.std(fw,axis=0)/(len(fw)**0.5)
    for m,s,clr in zip(mean.reshape(3,-1),sem.reshape(3,-1),'rgb'):
        ax.plot(x,m,color=clr)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)

x = np.arange(postTrials)+1
for ylbl in ('mice',):#'model'):
    for clust in np.unique(clustId):
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            for stim,clr,ls,lbl in zip(stimNames,'ggmm',('-','--','-','--'),('visual go','visual nogo','auditory go','auditory nogo')):
                c = 0
                resp = []
                for m in range(nMice):
                    block = 0
                    for i in range(nExps[m]):
                        if i<passSession[m]:
                            block += 5
                            continue
                        firstTrial = 0
                        for blockTrials in trialsPerBlock[m][i]:
                            trials = slice(firstTrial,firstTrial+blockTrials)
                            firstTrial += blockTrials
                            stimTrials = trialStim[m][i][trials]==stim
                            if trialRewardStim[m][i][trials][0]==rewardStim:
                                if clustId[c]==clust:
                                    if ylbl=='mice':
                                        r = Y[m][i][trials]
                                    else:
                                        r = prediction[holdOut][m][block]
                                    r = r[stimTrials][:postTrials]
                                    resp.append(np.full(postTrials,np.nan))
                                    resp[-1][:r.size] = r
                            block += 1
                            c += 1
                m = np.nanmean(resp,axis=0)
                s = np.nanstd(resp)/(len(resp)**0.5)
                ax.plot(x,m,color=clr,ls=ls,label=lbl)
                ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([0.5,postTrials+0.5])
            ax.set_ylim([0,1.01])
            ax.set_xlabel('Trials after block switch autorewards')
            ax.set_ylabel('Response rate of '+ylbl)
            ax.legend(loc='lower right')
            ax.set_title('cluster '+str(clust)+', '+blockLabel+' (n='+str(len(resp))+')')
            plt.tight_layout()







































