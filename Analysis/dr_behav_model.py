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
nTrialsPrev = 20
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
    
    
# plot example sessions
nExpsToPlot = 6
smoothSigma = 5
for m,exps in enumerate(expsByMouse):
    fig = plt.figure(figsize=(12,10))
    ylim = [-0.05,1.05]
    for i,obj in enumerate(exps[passSession[m]:passSession[m]+nExpsToPlot]):
        ax = fig.add_subplot(nExpsToPlot,1,i+1)
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock==blockInd+1
            blockStart,blockEnd = np.where(blockTrials)[0][[0,-1]]
            if goStim=='vis1':
                lbl = 'vis rewarded' if blockInd==0 else None
                ax.add_patch(matplotlib.patches.Rectangle([blockStart+0.5,ylim[0]],width=blockEnd-blockStart+1,height=ylim[1]-ylim[0],facecolor='0.8',edgecolor=None,alpha=0.2,zorder=0,label=lbl))
            for stim,clr,ls in zip(('vis1','vis2','sound1','sound2'),'ggmm',('-','--','-','--')):
                trials = blockTrials & (obj.trialStim==stim)
                smoothedRespProb = scipy.ndimage.gaussian_filter(obj.trialResponse[trials].astype(float),smoothSigma)
                lbl = stim if i==0 and blockInd==0 else None
                ax.plot(np.where(trials)[0]+1,smoothedRespProb,color=clr,ls=ls,label=lbl)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim([0.5,blockEnd+1.5])
        ax.set_ylim(ylim)
        if i==nExpsToPlot-1:
            ax.set_xlabel('Trial',fontsize=12)
        if i==0:
            ax.set_ylabel('Response rate',fontsize=12)
            ax.legend(bbox_to_anchor=(1,1.5),fontsize=8)
        # ax.set_title(obj.subjectName+'_'+obj.startTime,fontsize=10)
    plt.tight_layout()
    

# plot average response rate across blocks for mice and model
for holdOut in regressors:
    stimNames = ('vis1','vis2','sound1','sound2')
    postTrials = 15
    x = np.arange(postTrials)+1
    for ylbl in ('model',): # ('mice','model')
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            for stim,clr,ls,lbl in zip(stimNames,'ggmm',('-','--','-','--'),('visual go','visual nogo','auditory go','auditory nogo')):
                resp = []
                for m in range(nMice):
                    block = 0
                    for i in range(nExps[m]):
                        if i>4:
                            block += 5
                            continue
                        # if i<passSession[m]:
                        #     block += 5
                        #     continue
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

x = np.arange(nTrialsPrev)+1
for fw,title in zip((fwEarly,fwLate),('first 5 sessions','after performance criteria met')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    mean = np.mean(fw,axis=0)
    sem = np.std(fw,axis=0)/(len(fw)**0.5)
    for m,s,clr,lbl in zip(mean.reshape(3,-1),sem.reshape(3,-1),'rgb',regressors):
        ax.plot(x,m,color=clr,label=lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0.5,nTrialsPrev+0.5])
    ax.set_ylim([-0.15,0.8])
    ax.set_xlabel('Trials previous')
    ax.set_ylabel('Feature weight')
    ax.legend(title='features')
    ax.set_title(title)
    plt.tight_layout()

for ylbl,ylim in zip(('Mean','Max'),((0,0.4),(0,0.8))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(3)
    for fw,clr,lbl in zip((fwEarly,fwLate),'kc',('first 5 sessions','after perforamance criteria met')):
        y = np.mean(fw,axis=0).reshape(3,-1)
        y = y.mean(axis=1) if ylbl=='Mean' else y.max(axis=1)
        ax.plot(x,y,color=clr,lw=2,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(regressors)
    ax.set_xlim([-0.5,len(x)-0.5])
    ax.set_ylim(ylim)
    ax.set_ylabel(ylbl+' feature weight')
    ax.legend()
    plt.tight_layout()

for j,feature in enumerate(regressors):    
    for ylbl in ('Mean','Max'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        fw = np.full((nMice,max(nExps)),np.nan)
        for m,clr in enumerate(plt.cm.tab20(np.linspace(0,1,nMice))):
            f = []
            for w in featureWeights['none'][m]:
                w = w.reshape(3,-1)[j]
                w = w.mean() if ylbl=='Mean' else w.max()
                f.append(w)
            f = np.array(f).reshape(-1,5).mean(axis=1)
            ax.plot(np.arange(len(f))+1,f,color=clr,alpha=0.25)
            fw[m,:len(f)] = f
        m = np.nanmean(fw,axis=0)
        ax.plot(np.arange(len(m))+1,m,'k',lw=2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,len(m)+1])
        ax.set_xlabel('Session')
        ax.set_ylabel(ylbl+' feature weight')
        ax.set_title(ylbl+' '+feature+' weight')
        plt.tight_layout()


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

clustColors = 'mrkbc'

clustId,linkageMat = cluster(clustData,nClusters=5,plot=True,colors=clustColors,labels='off')

clustId,linkageMat = cluster(clustData,nClusters=5)


x = np.arange(nTrialsPrev)+1
for clust in np.unique(clustId):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fw = clustData[clustId==clust]
    mean = np.mean(fw,axis=0)
    sem = np.std(fw,axis=0)/(len(fw)**0.5)
    for m,s,clr,lbl in zip(mean.reshape(3,-1),sem.reshape(3,-1),'rgb',regressors):
        ax.plot(x,m,color=clr,label=lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0.5,nTrialsPrev+0.5])
    ax.set_ylim([-0.2,1.1])
    ax.set_xlabel('Trials previous')
    ax.set_ylabel('Feature weight')
    ax.legend(title='features')
    ax.set_title('Cluster '+str(clust))
    plt.tight_layout()
    
for ylbl,ylim in zip(('Mean','Max'),((-0.1,0.4),(0,0.9))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(3)
    for clust,clr in zip(np.unique(clustId),clustColors):
        fw = clustData[clustId==clust] 
        y = np.mean(fw,axis=0).reshape(3,-1)
        y = y.mean(axis=1) if ylbl=='Mean' else y.max(axis=1)
        ax.plot(x,y,color=clr,lw=2,label='Cluster '+str(clust))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(x)
    ax.set_xticklabels(regressors)
    ax.set_xlim([-0.5,len(x)-0.5])
    ax.set_ylim(ylim)
    ax.set_ylabel(ylbl+' feature weight')
    ax.legend(loc='upper left')
    plt.tight_layout()
        
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
            ax.set_title('Cluster '+str(clust)+', '+blockLabel+' (n='+str(len(resp))+')')
            plt.tight_layout()


# cluster raw behavior data

def pca(data,plot=False):
    # data is n samples x m parameters
    eigVal,eigVec = np.linalg.eigh(np.cov(data,rowvar=False))
    order = np.argsort(eigVal)[::-1]
    eigVal = eigVal[order]
    eigVec = eigVec[:,order]
    pcaData = data.dot(eigVec)
    if plot:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(1,eigVal.size+1),eigVal.cumsum()/eigVal.sum(),'k')
        ax.set_xlim((0.5,eigVal.size+0.5))
        ax.set_ylim((0,1.02))
        ax.set_xlabel('PC')
        ax.set_ylabel('Cumulative Fraction of Variance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(eigVec,clim=(-1,1),cmap='bwr',interpolation='none',origin='lower')
        ax.set_xlabel('PC')
        ax.set_ylabel('Parameter')
        ax.set_title('PC Weightings')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(length=0)
        cb.set_ticks([-1,0,1])
    return pcaData,eigVal,eigVec


respProb = {stim: [] for stim in stimNames}
smoothSigma = 5
tintp = np.arange(600)
for m,exps in enumerate(expsByMouse):
    for i,obj in enumerate(exps[passSession[m]:]):
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock==blockInd+1
            for stim,clr,ls in zip(('vis1','vis2','sound1','sound2'),'ggmm',('-','--','-','--')):
                trials = blockTrials & (obj.trialStim==stim) & ~obj.autoRewarded
                startTime = obj.trialStartTimes[trials]
                startTime = startTime-startTime[0]
                r = scipy.ndimage.gaussian_filter(obj.trialResponse[trials].astype(float),smoothSigma)
                respProb[stim].append(np.interp(tintp,startTime,r))

clustData = np.concatenate([respProb[stim] for stim in respProb],axis=1)

pcaData,eigVal,eigVec = pca(clustData,plot=True)

clustId,linkageMat = cluster(pcaData[:,:6],nClusters=5,plot=True,colors=clustColors,labels='off',nreps=1000)

for clust in np.unique(clustId):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
        r = np.array(respProb[stim])[clustId==clust]
        m = np.nanmean(r,axis=0)
        s = np.nanstd(r)/(len(r)**0.5)
        ax.plot(tintp,m,color=clr,ls=ls,label=stim)
        ax.fill_between(tintp,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response rate')
    ax.legend(loc='lower right')
    ax.set_title('Cluster '+str(clust))
    plt.tight_layout()




respProb = {goStim: {stim: [] for stim in stimNames} for goStim in ('vis1','sound1')}
smoothSigma = 5
tintp = np.arange(600)
for m,exps in enumerate(expsByMouse):
    for i,obj in enumerate(exps[passSession[m]:]):
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock==blockInd+1
            for stim,clr,ls in zip(('vis1','vis2','sound1','sound2'),'ggmm',('-','--','-','--')):
                trials = blockTrials & (obj.trialStim==stim) & ~obj.autoRewarded
                startTime = obj.trialStartTimes[trials]
                startTime = startTime-startTime[0]
                r = scipy.ndimage.gaussian_filter(obj.trialResponse[trials].astype(float),smoothSigma)
                respProb[goStim][stim].append(np.interp(tintp,startTime,r))

vis = np.concatenate([respProb['vis1'][stim] for stim in stimNames],axis=1)
sound = np.concatenate([respProb['sound1'][stim] for stim in ('sound1','sound2','vis1','vis2')],axis=1)
clustData = np.concatenate((vis,sound),axis=0)

pcaData,eigVal,eigVec = pca(clustData,plot=True)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for pc in range(6):
    ax.plot(eigVec[:,pc]*eigVal[pc])
    

clustId,linkageMat = cluster(pcaData[:,:6],nClusters=5,plot=True,colors=clustColors,labels='off',nreps=1000)

for clust in np.unique(clustId):
    for j,(rewardStim,blockLabel) in enumerate(zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks'))):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
            r = np.array(respProb[rewardStim][stim])[clustId[len(vis)*j:len(vis)*(j+1)]==clust]
            m = np.nanmean(r,axis=0)
            s = np.nanstd(r)/(len(r)**0.5)
            ax.plot(tintp,m,color=clr,ls=ls,label=stim)
            ax.fill_between(tintp,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response rate')
        ax.legend(loc='lower right')
        ax.set_title('Cluster '+str(clust)+', '+blockLabel+' (n='+str(len(r))+')')
        plt.tight_layout()






















