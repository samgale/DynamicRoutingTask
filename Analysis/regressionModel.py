# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:39:53 2023

@author: svc_ccg
"""

import copy, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from DynamicRoutingAnalysisUtils import getFirstExperimentSession,getSessionsToPass,getSessionData



baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

miceToIgnore = summaryDf['wheel fixed'] & summaryDf['cannula']

hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])

sessionData = {lbl: [] for lbl in ('training','nogo','noAR','rewardOnly','no reward')}

ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['stage 5 repeats'] & ~miceToIgnore
mice = np.array(summaryDf[ind]['mouse id'])
sessionsToPass = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
    firstExperimentSession = getFirstExperimentSession(df)
    if firstExperimentSession is not None:
        sessions[firstExperimentSession:] = False
    sessions = np.where(sessions)[0]
    sessionsToPass.append(getSessionsToPass(mid,df,sessions,stage=5))
    sessionData['training'].append([getSessionData(mid,startTime) for startTime in df.loc[sessions,'start time']])

mice = {'nogo': np.array(summaryDf[summaryDf['nogo']]['mouse id']),
        'noAR': np.array(summaryDf[summaryDf['noAR']]['mouse id']),
        'rewardOnly': np.array(summaryDf[summaryDf['rewardOnly']]['mouse id']),
        'no reward': np.array(summaryDf[summaryDf['no reward']]['mouse id'])}
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array([lbl in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
        sessionData[lbl].append([getSessionData(mid,startTime) for startTime in df.loc[sessions,'start time']])
        

# construct regressors
trainingPhases = ('initial training','after learning') + tuple(sessionData.keys())[1:]
nTrialsPrev = 20
regressors = ('reinforcement','posReinforcement','negReinforcement',
              'crossModalReinforcement','crossModalPosReinforcement','crossModalNegReinforcement',
              'perseveration','reward')

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
    regData[phase]['trialResponseTime'] = []
    regData[phase]['X'] = []
    s = -1
    b = -1
    for m,exps in enumerate((sessionData['training'] if phase in trainingPhases[:2] else sessionData[phase])):
        if phase == 'initial training':
            exps = exps[:5]
        elif phase == 'after learning':
            exps = exps[sessionsToPass[m]:]
        for sn,obj in enumerate(exps):
            print(phase,m,sn)
            s += 1
            respTimes = np.full(obj.responseTimes.size,np.nan)
            for stim in ('vis1','sound1'):
                i = obj.trialStim == stim
                respTimes[i] = (obj.responseTimes[i] - np.nanmean(obj.responseTimes[i])) / np.nanstd(obj.responseTimes[i])
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
                for r in regressors:
                    regData[phase]['X'][-1][r] = np.zeros((nTrials,nTrialsPrev))
                    for n in range(1,nTrialsPrev+1):
                        for trial,stim in enumerate(obj.trialStim[trials]):
                            resp = obj.trialResponse[:trialInd[trial]]
                            rew = obj.trialRewarded[:trialInd[trial]]
                            trialStim = obj.trialStim[:trialInd[trial]]
                            sameStim = trialStim==stim
                            otherModalTarget = 'vis1' if stim[:-1]=='sound' else 'sound1'
                            otherModal = trialStim==otherModalTarget
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
                            elif r=='perseveration' and sameStim[-n] and resp[-n]:
                                regData[phase]['X'][-1][r][trial,n-1] = 1
                            # if r=='reinforcement' and sameStim.sum() > n:
                            #     regData[phase]['X'][-1][r][trial,n-1] = 1 if rew[sameStim][-n] else (-1 if resp[sameStim][-n] else 0)
                            # elif r=='posReinforcement' and sameStim.sum() > n and rew[sameStim][-n]:
                            #     regData[phase]['X'][-1][r][trial,n-1] = 1
                            # elif r=='negReinforcement' and sameStim.sum() > n and resp[sameStim][-n] and not rew[sameStim][-n]:
                            #     regData[phase]['X'][-1][r][trial,n-1] = 1
                            # elif r=='crossModalReinforcement' and otherModal.sum() > n:
                            #     regData[phase]['X'][-1][r][trial,n-1] = 1 if rew[otherModal][-n] else (-1 if resp[otherModal][-n] else 0)
                            # elif r=='crossModalPosReinforcement' and otherModal.sum() > n and rew[otherModal][-n]:
                            #     regData[phase]['X'][-1][r][trial,n-1] = 1
                            # elif r=='crossModalNegReinforcement' and otherModal.sum() > n and resp[otherModal][-n] and not rew[otherModal][-n]:
                            #     regData[phase]['X'][-1][r][trial,n-1] = 1
                            # elif r=='perseveration' and sameStim.sum() > n and resp[sameStim][-n]:
                            #     regData[phase]['X'][-1][r][trial,n-1] = 1   
                            elif r=='reward' and rew[-n]:
                                regData[phase]['X'][-1][r][trial,n-1] = 1
                regData[phase]['mouseIndex'].append(m)
                regData[phase]['sessionIndex'].append(s)
                regData[phase]['blockIndex'].append(b)
                regData[phase]['blockNumber'].append(blockInd+1)
                regData[phase]['sessionNumber'].append(sn+1)
                regData[phase]['rewardStim'].append(obj.blockStimRewarded[blockInd])
                regData[phase]['trialStim'].append(obj.trialStim[trials])
                regData[phase]['trialResponse'].append(obj.trialResponse[trials])
                regData[phase]['trialResponseTime'].append(respTimes[trials])
                
                
# regressor correlations
fitRegressors = ('posReinforcement','negReinforcement','crossModalPosReinforcement','crossModalNegReinforcement','perseveration','reward')
for phase in ('after learning',):#in trainingPhases:
    mi = np.array(regData[phase]['mouseIndex'])
    si = np.array(regData[phase]['sessionIndex'])
    r = []
    for m in np.unique(mi):
        r.append(np.zeros((len(fitRegressors),)*2))
        c = np.zeros(r[-1].shape)
        for s in np.unique(si[mi==m]):
            for b in np.where(si==s)[0]:
                for i,r1 in enumerate(fitRegressors):
                    for j,r2 in enumerate(fitRegressors):
                        r[-1][i,j] += np.corrcoef(*(regData[phase]['X'][b][d].flatten() for d in (r1,r2)))[0,1]
                        c[i,j] += 1
        r[-1] /= c
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(np.nanmean(r,axis=0))
    plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    break
        

# fit model
fitType = 'response'
fitRegressors = ('posReinforcement','negReinforcement','crossModalPosReinforcement','crossModalNegReinforcement','reward')
holdOutRegressor = ('none',) #+ fitRegressors #+ (('reinforcement','crossModalReinforcement'),('reward','action'))
regressorColors = ([s for s in 'grmbkcy']+['0.5'])[:len(fitRegressors)]

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
                    y[-1].append(regData[phase][('trialResponse' if fitType=='response' else 'trialResponseTime')][b])
                ntrials.append([len(b) for b in x[-1]])
                x[-1] = np.concatenate(x[-1])
                y[-1] = np.concatenate(y[-1])
            for i in range(len(x)):
                trainX = np.concatenate(x[:i]+x[i+1:])
                trainY = np.concatenate(y[:i]+y[i+1:])
                testX = x[i]
                testY = y[i]
                if fitType == 'response':
                    model = LogisticRegression(C=1.0,max_iter=1e3)
                else:
                    model = Ridge(alpha=1)
                notNan = ~np.isnan(trainY)
                model.fit(trainX[notNan],trainY[notNan])
                trainAccuracy[phase][h][-1].append(model.score(trainX[notNan],trainY[notNan]))
                notNan = ~np.isnan(testY)
                accuracy[phase][h][-1].append(model.score(testX[notNan],testY[notNan]))
                pred = np.full(testY.size,np.nan)
                pred[notNan] = model.predict(testX[notNan])
                if fitType == 'response':
                    balancedAccuracy[phase][h][-1].append(sklearn.metrics.balanced_accuracy_score(testY,pred))
                    predProb = model.predict_proba(testX)[:,1]
                    logLoss[phase][h][-1].append(sklearn.metrics.log_loss(testY,predProb,labels=(True,False)))
                featureWeights[phase][h][-1].append(np.squeeze(model.coef_))
                bias[phase][h][-1].append(model.intercept_)
                nstart = 0
                for n in ntrials[i]:
                    prediction[phase][h].append(pred[nstart:nstart+n])
                    if fitType == 'response':
                        predictionProb[phase][h].append(predProb[nstart:nstart+n])
                    nstart += n
    

# plots
for phase in trainingPhases:
    fig = plt.figure(figsize=(5,8))
    ax = fig.add_subplot(1,1,1)
    for x,h in enumerate(holdOutRegressor):
        d = [np.nanmean((np.array(a)-np.array(b))/np.array(a)) for a,b in zip(balancedAccuracy[phase]['none'],balancedAccuracy[phase][h])]
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
for phase in trainingPhases:
    for h in holdOutRegressor:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # d = [np.mean(b) for b in bias[phase][h]]
        # m = np.mean(d)
        # s = np.std(d)/(len(d)**0.5)
        # ax.plot([x[0],x[-1]],[m,m],color='0.7')
        # ax.fill_between([x[0],x[-1]],[m+s]*2,[m-s]*2,color='0.7',alpha=0.25)
        reg,clrs = zip(*[(r,c) for r,c in zip(fitRegressors,regressorColors) if r!=h and r not in h])
        d = [np.mean(fw,axis=0) for fw in featureWeights[phase][h]]
        mean = np.mean(d,axis=0)
        sem = np.std(d,axis=0)/(len(d)**0.5)
        for m,s,clr,lbl in zip(mean.reshape(len(reg),-1),sem.reshape(len(reg),-1),clrs,reg):
            ax.plot(x,m,color=clr,label=lbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks([1,5,10,15,20])
        ax.set_xlim([0.5,nTrialsPrev+0.5])
        # ax.set_ylim([-0.15,0.8])
        ax.set_xlabel('Trials previous')
        ax.set_ylabel('Regression weight')
        ax.legend(title='features',loc='upper right')
        ax.set_title(phase+', '+h)
        plt.tight_layout()


for phase in trainingPhases:
    for h in holdOutRegressor:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot([-1,10],[0,0],'--',color='0.5')
        reg,clrs = zip(*[(r,c) for r,c in zip(fitRegressors,regressorColors) if r!=h and r not in h])
        d = [np.mean([np.sum(np.reshape(w,(len(reg),-1)),axis=1) for w in fw],axis=0) for fw in featureWeights[phase][h]]
        b = [np.mean(b) for b in bias[phase][h]]
        mean = np.concatenate((np.mean(d,axis=0),[np.mean(b)]))
        sem = np.concatenate((np.std(d,axis=0)/(len(d)**0.5),[np.std(b)/(len(b)**0.5)]))
        for x,(m,s)in enumerate(zip(mean,sem)):
            ax.plot(x,m,'o',mec='k',mfc='none')
            ax.plot([x,x],[m-s,m+s],'k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(len(reg)+1))
        ax.set_xticklabels(reg+('bias',))
        ax.set_xlim([-0.5,len(reg)+0.5])
        ax.set_ylabel('Sum of regression weights')
        ax.set_title(phase+', '+h)
        plt.tight_layout()


for phase in trainingPhases:
    for h in holdOutRegressor:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot([-1,10],[0,0],'--',color='0.5')
        reg,clrs = zip(*[(r,c) for r,c in zip(fitRegressors,regressorColors) if r!=h and r not in h])
        d = [np.mean([np.reshape(w,(len(reg),-1))[:,0] for w in fw],axis=0) for fw in featureWeights[phase][h]]
        b = [np.mean(b) for b in bias[phase][h]]
        mean = np.concatenate((np.mean(d,axis=0),[np.mean(b)]))
        sem = np.concatenate((np.std(d,axis=0)/(len(d)**0.5),[np.std(b)/(len(b)**0.5)]))
        for x,(m,s)in enumerate(zip(mean,sem)):
            ax.plot(x,m,'o',mec='k',mfc='none')
            ax.plot([x,x],[m-s,m+s],'k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(len(reg)+1))
        ax.set_xticklabels(reg+('bias',))
        ax.set_xlim([-0.5,len(reg)+0.5])
        ax.set_ylabel('Previous trial regression weight')
        ax.set_title(phase+', '+h)
        plt.tight_layout()


postTrials = 15
x = np.arange(postTrials)+1
for phase in trainingPhases:
    for h in holdOutRegressor:
        fig = plt.figure()
        for i,(d,ylbl) in enumerate(zip((regData[phase][('trialResponse' if fitType=='response' else 'trialResponseTime')],prediction[phase][h]),('mice','model'))):
            ax = fig.add_subplot(2,1,i+1)
            for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
                y = []
                for m in np.unique(regData[phase]['mouseIndex']):
                    resp = []
                    for j,r in enumerate(d): #range(len(regData['blockIndex'])):
                        if regData[phase]['mouseIndex'][j]==m:
                            rewStim = regData[phase]['rewardStim'][j]
                            nonRewStim = np.setdiff1d(('vis1','sound1'),rewStim)
                            stim =  nonRewStim if 'unrewarded' in stimLbl else rewStim
                            resp.append(np.full(postTrials,np.nan))
                            a = r[regData[phase]['trialStim'][j]==stim][:postTrials]
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
            # ax.set_ylim([0,1.01])
            if i==1:
                ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
            ax.set_ylabel('Response rate of '+ylbl,fontsize=12)
            if i==0:
                ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        ax.set_title(h)
        plt.tight_layout()










