# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:55:19 2022

@author: svc_ccg
"""

import copy
import os
import re
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData,sortExps
import sklearn
from sklearn.linear_model import LogisticRegression
import psytrack


def crossValidate(model,X,y,nSplits):
    # cross validation using stratified shuffle split
    # each split preserves the percentage of samples of each class
    # all samples used in one test set
    
    classVals = np.unique(y)
    nClasses = len(classVals)
    nSamples = len(y)
    samplesPerClass = [np.sum(y==val) for val in classVals]
    samplesPerSplit = [round(n/nSplits) for n in samplesPerClass]
    shuffleInd = np.random.permutation(nSamples)
    
    cv = {'estimator': [sklearn.base.clone(model) for _ in range(nSplits)]}
    cv['train_score'] = []
    cv['test_score'] = []
    cv['predict'] = np.full(nSamples,np.nan)
    cv['predict_proba'] = np.full((nSamples,nClasses),np.nan)
    cv['decision_function'] = np.full((nSamples,nClasses),np.nan) if nClasses>2 else np.full(nSamples,np.nan)
    cv['feature_importance'] = []
    cv['coef'] = []
    modelMethods = dir(model)
    
    for k,estimator in enumerate(cv['estimator']):
        testInd = []
        for val,n in zip(classVals,samplesPerSplit):
            start = k*n
            ind = shuffleInd[y[shuffleInd]==val]
            testInd.extend(ind[start:start+n] if k+1<nSplits else ind[start:])
        trainInd = np.setdiff1d(shuffleInd,testInd)
        estimator.fit(X[trainInd],y[trainInd])
        cv['train_score'].append(estimator.score(X[trainInd],y[trainInd]))
        cv['test_score'].append(estimator.score(X[testInd],y[testInd]))
        cv['predict'][testInd] = estimator.predict(X[testInd])
        for method in ('predict_proba','decision_function'):
            if method in modelMethods:
                cv[method][testInd] = getattr(estimator,method)(X[testInd])
        for attr in ('feature_importance_','coef_'):
            if attr in estimator.__dict__:
                cv[attr[:-1]].append(getattr(estimator,attr))
    return cv


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"

mouseIds = ('638573','638574','638575','638576','638577','638578')

excelPath = os.path.join(baseDir,'DynamicRoutingTraining.xlsx')
sheets = pd.read_excel(excelPath,sheet_name=None)
allMiceDf = sheets['all mice']

mice = []
sessionStartTimes = []
for mid in mouseIds:
    mouseInd = np.where(allMiceDf['mouse id']==int(mid))[0][0]
    df = sheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']])
    if sessions.sum() > 0:
        mice.append(str(mid))
        sessionStartTimes.append(list(df['start time'][sessions]))
        
expsByMouse = []
for mid,st in zip(mice,sessionStartTimes):
    expsByMouse.append([])
    for t in st:
        f = os.path.join(baseDir,'Data',mid,'DynamicRouting1_' + mid + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5')
        obj = DynRoutData()
        obj.loadBehavData(f)
        expsByMouse[-1].append(obj)

for exps in expsByMouse:
    dprimeSame = np.full((len(exps),6),np.nan)
    dprimeOther = dprimeSame.copy()
    for i,obj in enumerate(exps):
        dprimeSame[i] = obj.dprimeSameModal
        dprimeOther[i] = obj.dprimeOtherModalGo


nTrialsPrev = 10
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
x = [{r: [] for r in regressors} for _ in range(nMice)]
y = copy.deepcopy(trialsPerSession)
for m,exps in enumerate(expsByMouse):
    for obj in exps:
        trials = ~obj.catchTrials & ~obj.autoRewarded & (obj.trialBlock>1)
        trialInd = np.where(trials)[0]
        nTrials = trials.sum()
        for r in regressors:
            x[m][r].append(np.zeros((nTrials,nTrialsPrev)))
            for n in range(1,nTrialsPrev+1):
                for trial,stim in enumerate(obj.trialStim[trials]):
                    if r in ('reinforcement','persistence'):
                        sameStim = obj.trialStim[:trialInd[trial]] == stim
                        if sameStim.sum()>n:  
                            if r=='reinforcement':
                                if obj.trialResponse[:trialInd[trial]][sameStim][-n]:
                                    x[m][r][-1][trial,n-1] = 1 if obj.trialRewarded[:trialInd[trial]][sameStim][-n] else -1
                            elif r=='persistence':
                                x[m][r][-1][trial,n-1] = obj.trialResponse[:trialInd[trial]][sameStim][-n]
                    elif r=='attention':
                        notCatch = obj.trialStim[:trialInd[trial]] != 'catch'
                        if notCatch.sum()>n:
                            if obj.trialRewarded[:trialInd[trial]][notCatch][-n]:
                                sameModal = any(s in stim and s in obj.trialStim[:trialInd[trial]][notCatch][-n] for s in ('vis','sound'))
                                x[m][r][-1][trial,n-1] = 1 if sameModal else -1
        y[m].append(obj.trialResponse[trials].astype(float))
        trialsPerSession[m].append(nTrials)
        trialsPerBlock[m].append([np.sum(obj.trialBlock[trials]==block) for block in np.unique(obj.trialBlock[trials])])
        trialStim[m].append(obj.trialStim[trials])
        trialRewardStim[m].append(obj.rewardedStim[trials])



c = []
for m in range(nMice):
    for i in range(nExps[m]):
        firstTrial = 0
        for j,blockTrials in enumerate(trialsPerBlock[m][i]):
            print(m,i,j)
            X = np.concatenate([x[m][r][i][firstTrial:firstTrial+blockTrials] for r in regressors],axis=1)
            model = LogisticRegression(fit_intercept=True,max_iter=1e3)
            model.fit(X,y[m][i][firstTrial:firstTrial+blockTrials])
            c.append(model.coef_.flatten())
            firstTrial += blockTrials




    
# psytrack
hyperparams = []
evidence = []
modelWeights = []
hessian = []
cvLikelihood =[]
cvProbNoLick = []
accuracy = []
cvFolds = 2


for m in [0]:#range(nMice):
    for i in range(nExps[m]):
        print(m,i)
        d = {'inputs': {key: val[i] for key,val in x[m].items()},
             'y': y[m][i],
             'dayLength': trialsPerBlock[m][i]}
        
        weights = {key: val.shape[1] for key,val in d['inputs'].items()}
        
        nWeights = sum(weights.values())
        
        hyper= {'sigInit': 2**4.,
                'sigma': [2**-4.] * nWeights,
                'sigDay': [2**-4.] * nWeights}
        
        optList = ['sigma','sigDay']
        
        # try:
        t = time.perf_counter()
        hyp, evd, wMode, hess_info = psytrack.hyperOpt(d, hyper, weights, optList)
        hyperparams.append(hyp)
        evidence.append(evd)
        modelWeights.append(wMode)
        hessian.append(hess_info)
        print(time.perf_counter()-t)
        break
        # except:
        #     print('\nerror fitting ',m)
        #     hyperparams.append(None)
        #     evidence.append(np.nan)
        #     modelWeights.append(np.full((nWeights,d['y'].size),np.nan))
        #     hessian.append(None)
        
        if cvFolds is not None:
            cvTrials = d['y'].size - (d['y'].size % cvFolds)
            likelihood = np.nan
            probNoLick = np.full(cvTrials,np.nan)
            if hyperparams[-1] is not None:
                try:
                    likelihood,probNoLick = psytrack.crossValidate(psytrack.trim(d,END=cvTrials), hyper, weights, optList, F=cvFolds, seed=0)
                except:
                    print('\nerror cross validating ',m)
            cvLikelihood.append(likelihood)
            cvProbNoLick.append(probNoLick)
            d['y'] -= 1
            accuracy.append(np.abs(d['y'][:cvTrials] - probNoLick))



preTrials = postTrials = 45
x = np.arange(-preTrials,postTrials+1)
for ho in holdOut:
    title = 'all regressors' if ho=='none' else 'no '+ho+' regressor'
    for blockType,rewardStim in zip(('visual rewarded','auditory rewarded'),('vis1','sound1')):
        visGoRespProb = []
        visNogoRespProb = []
        soundGoRespProb = []
        soundNogoRespProb = []
        weights = []
        for i in range(len(exps)):
            blockStartStop = np.cumsum(sessionBlockTrials[i])
            lickProb = 1-cvProbNoLick[ho][i]
            mWeights = modelWeights[ho][i]
            for j,(blockStart,blockStop) in enumerate(zip(blockStartStop[:-1],blockStartStop[1:])):
                if sessionRewardStim[i][blockStart]==rewardStim:
                    for rp,stim in zip((visGoRespProb,visNogoRespProb,soundGoRespProb,soundNogoRespProb),('vis1','vis2','sound1','sound2')):
                        stimTrials = sessionStim[i][:lickProb.size]==stim
                        r = np.full(preTrials+postTrials+1,np.nan)
                        r[:preTrials] = lickProb[:blockStart][stimTrials[:blockStart]][-preTrials:]
                        post = lickProb[blockStart:][stimTrials[blockStart:]][:postTrials]
                        r[preTrials+1:preTrials+1+post.size] = post
                        rp.append(r)
                    w = np.full((mWeights.shape[0],preTrials+postTrials+1),np.nan)
                    for k,mw in enumerate(mWeights):
                        w[k,:preTrials] = mw[blockStart-preTrials:blockStart]
                        w[k,preTrials+1:] = mw[blockStart:blockStart+postTrials]
                    weights.append(w)
            
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ylim = [0,1.01]
        ax.plot([0,0],ylim,'k--')
        for d,clr,ls,lbl in zip((visGoRespProb,visNogoRespProb,soundGoRespProb,soundNogoRespProb),'ggmm',('-','--','-','--'),('visual go','visual nogo','auditory go','auditory nogo')):
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=lbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim(ylim)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Cross-Validated Response Probability')
        ax.set_title(title+'\n'+blockType)
        # ax.legend(loc='lower right')
        plt.tight_layout()
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        wNames = sorted([r for r in regressors if r!=ho and r not in ho])
        wColors = [regressorColors[regressors.index(wn)] for wn in wNames]
        wMean = np.nanmean(weights,axis=0)
        wSem = np.nanstd(weights,axis=0)/(np.sum(~np.isnan(weights),axis=0)**0.5)
        for m,s,clr,lbl in zip(wMean,wSem,wColors,wNames):
            lbl = 'preservation' if lbl=='persistence' else lbl
            ax.plot(x,m,clr,label=lbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Regressor Weight')
        ax.set_title(title+'\n'+blockType)
        ax.legend(loc='lower right')
        plt.tight_layout()


for m,lbl in zip((evidence,cvLikelihood,accuracy),('evidence','likelihood','accuracy')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for reg,clr in zip(holdOut[1:],holdOutColors[1:]):
        if lbl=='accuracy':
            a = np.array([np.mean(b) for b in m['none']])
            h = np.array([np.mean(d) for d in m[reg]])
        else:
            a = np.array(m['none'])
            h = np.array(m[reg])
        d = h-a
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=reg)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('change in model '+lbl,fontsize=12)
    ax.set_ylabel('cumulative prob',fontsize=12)
    ax.legend()
    plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for reg,clr in zip(holdOut,holdOutColors):
    d = np.array([np.mean(a) for a in accuracy[reg]])
    dsort = np.sort(d)
    cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
    ax.plot(dsort,cumProb,color=clr,label=reg)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_ylim([0,1.02])
ax.set_xlabel('accuracy',fontsize=12)
ax.set_ylabel('cumulative prob',fontsize=12)
ax.legend()
plt.tight_layout()
    

for ho in holdOut:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    r,c = zip(*[(r,c) for r,c in zip(regressors,regressorColors) if r!=ho and r not in ho])
    wNames = sorted(r)
    for reg,clr in zip(r,c):
        d = np.array([np.nanmean(w[wNames.index(reg)]) for w in modelWeights[ho]])
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=reg)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('regressor weight',fontsize=12)
    ax.set_ylabel('cumulative fraction of sessions',fontsize=12)
    # ax.set_title(ho)
    ax.legend()
    plt.tight_layout()
    

for ho in holdOut:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    r,c = zip(*[(r,c) for r,c in zip(regressors,regressorColors) if r!=ho and r not in ho])
    wNames = sorted(r)
    for reg,clr in zip(r,c):
        d = []
        for i,w in enumerate(modelWeights[ho]):
            blockStartStop = np.concatenate(([0],np.cumsum(sessionBlockTrials[i])))
            for blockStart,blockStop in zip(blockStartStop[:-1],blockStartStop[1:]):
                d.append(np.nanmean(w[wNames.index(reg)][blockStart:blockStop]))
        d = np.array(d)
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=reg)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('regressor weight',fontsize=12)
    ax.set_ylabel('cumulative fraction of blocks',fontsize=12)
    # ax.set_title(ho)
    ax.legend()
    plt.tight_layout()
    

for ho in holdOut:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    r,c = zip(*[(r,c) for r,c in zip(regressors,regressorColors) if r!=ho and r not in ho])
    wNames = sorted(r)
    for reg,clr in zip(r,c):
        d = np.array([np.nanmean(np.concatenate([modelWeights[ho][i][wNames.index(reg)] for i in np.where(mouseIds==mid)[0]])) for mid in np.unique(mouseIds)])
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=reg)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('model weighting',fontsize=12)
    ax.set_ylabel('cumulative fraction of mice',fontsize=12)
    # ax.set_title(ho)
    ax.legend()
    plt.tight_layout()


alim = [-4,14.5]
for ho in ('none',):#holdOut:
    reg = [r for r in regressors if r!=ho and r not in ho]
    wNames = sorted(reg)    
    if 'model' in wNames and 'reinforcement' in wNames:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        mod,reinf = [[np.mean(w[wNames.index(r)]) for w in modelWeights[ho]] for r in ('model','reinforcement')]
        ax.plot(mod,reinf,'ko',label='sessions')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_xlabel('model regressor weight',fontsize=12)
        ax.set_ylabel('reinforcment regressor weight',fontsize=12)
        ax.legend()
        # ax.set_title(ho)
        plt.tight_layout()
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        mod = []
        reinf = []
        modFirst = []
        reinfFirst = []
        for reg,d,df in zip(('model','reinforcement'),(mod,reinf),(modFirst,reinfFirst)):
            for i,w in enumerate(modelWeights[ho]):
                blockStartStop = np.concatenate(([0],np.cumsum(sessionBlockTrials[i])))
                for j,(blockStart,blockStop) in enumerate(zip(blockStartStop[:-1],blockStartStop[1:])):
                    d.append(np.nanmean(w[wNames.index(reg)][blockStart:blockStop]))
                    if j==0:
                        df.append(d[-1])
        ax.plot(mod,reinf,'ko',label='blocks')
        ax.plot(modFirst,reinfFirst,'ro',label='first blocks')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim(alim)
        ax.set_ylim(alim)
        ax.set_xlabel('model regressor weight',fontsize=12)
        ax.set_ylabel('reinforcment regressor weight',fontsize=12)
        # ax.set_title(ho)
        ax.legend()
        plt.tight_layout()
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        mod,reinf = [[np.nanmean(np.concatenate([modelWeights[ho][i][wNames.index(reg)] for i in np.where(mouseIds==mid)[0]])) for mid in np.unique(mouseIds)] for reg in ('model','reinforcement')]
        ax.plot(mod,reinf,'ko')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlabel('model weight',fontsize=12)
        ax.set_ylabel('reinforcment weight',fontsize=12)
        # ax.set_title(ho)
        plt.tight_layout()
        

for ho in holdOut:
    reg = [r for r in regressors if r!=ho and r not in ho]
    wNames = sorted(reg)
    wCorr = np.full((len(reg),)*2,np.nan)
    for i,ri in enumerate(reg):
        wi = np.array([np.mean(w[wNames.index(ri)]) for w in modelWeights[ho]])
        for j,rj in enumerate(reg):
            if i!=j:
                wj = np.array([np.mean(w[wNames.index(rj)]) for w in modelWeights[ho]])
                ind = (~np.isnan(wi)) & (~np.isnan(wj))
                wCorr[i,j] = np.corrcoef(wi[ind],wj[ind])[0,1]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    cmax = 1 #np.nanmax(np.absolute(wCorr))
    im = ax.imshow(wCorr,clim=[-cmax,cmax],cmap='bwr')
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(len(wNames)))
    ax.set_yticks(np.arange(len(wNames)))
    ax.set_xlim([-0.5,len(wNames)-0.5])
    ax.set_ylim([len(wNames)-0.5,-0.5])
    ax.set_xticklabels(wNames)
    ax.set_yticklabels(wNames)
    ax.set_title(ho)
    cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
    plt.tight_layout()
        
        
        


    
for i in range(len(exps)):
    w = modelWeights['none'][i]
    wNames = sorted(regressors)
    wColors = [regressorColors[regressors.index(wn)] for wn in wNames]
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ylim = [min(0,1.05*w.min()),1.05*w.max()]
    for blockEnd in np.cumsum(sessionBlockTrials[i])[:-1]:
        ax.plot([blockEnd+0.5]*2,ylim,'k')
    for w,lbl,clr in zip(w,wNames,wColors):
        ax.plot(np.arange(sessionTrials[i])+1,w,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,sessionTrials[i]+1])
    ax.set_ylim(ylim)
    ax.set_xlabel('trial',fontsize=12)
    ax.set_ylabel('weights',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1),fontsize=8)
    ax.set_title(exps[i].subjectName+'_'+exps[i].startTime,fontsize=10)
    plt.tight_layout()


smoothSigma = 5
for i in range(len(exps)):
    lickProbModel= 1-cvProbNoLick['none'][i]
    fig = plt.figure(figsize=(8,8))
    ylim = [-0.05,1.05]
    ax = fig.add_subplot(1,1,1)
    for j,(stim,clr) in enumerate(zip(('vis1','vis2','sound1','sound2'),'rmbc')):
        stimInd = sessionStim[i][:lickProbModel.size] == stim
        blockStart = 0
        smoothedLickProb = []
        smoothedLickProbModel = []
        for blockEnd in np.cumsum(sessionBlockTrials[i]):
            if j==0:
                ax.plot([blockEnd+0.5]*2,ylim,'k')
            blockInd = slice(blockStart,blockEnd)
            trialInd = stimInd[blockInd]
            smoothedLickProb.append(gaussian_filter(y[i].astype(float)[:lickProbModel.size][blockInd][trialInd],smoothSigma))
            smoothedLickProbModel.append(gaussian_filter(lickProbModel[blockInd][trialInd],smoothSigma))
            blockStart = blockEnd
        trials = np.where(stimInd)[0]+1
        ax.plot(trials,np.concatenate(smoothedLickProb),color=clr,label=stim+' (mouse, smoothed)')
        ax.plot(trials,np.concatenate(smoothedLickProbModel),'--',color=clr,label=stim+' (model, smoothed)')
        ax.plot(trials,lickProbModel[stimInd],'o',ms=2,mec=clr,mfc='none',label=stim+' (model)')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0,sessionTrials[i]+1])
    ax.set_ylim(ylim)
    ax.set_xlabel('trial',fontsize=12)
    ax.set_ylabel('resp prob',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1.5),fontsize=8)
    ax.set_title(exps[i].subjectName+'_'+exps[i].startTime,fontsize=10)
    plt.tight_layout()





















