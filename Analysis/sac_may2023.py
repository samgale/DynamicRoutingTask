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
from statsmodels.stats.multitest import multipletests


# get data
baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"

excelPath = os.path.join(baseDir,'DynamicRoutingTraining.xlsx')
sheets = pd.read_excel(excelPath,sheet_name=None)
allMiceDf = sheets['all mice']

mouseIds = allMiceDf['mouse id']
passOnly = True

mouseIds = ('638573','638574','638575','638576','638577','638578',
            '649943','653481','656726')
passOnly = False

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

    
# running
fig = plt.figure(figsize=(8,8))
fig.suptitle(str(nMice)+' mice')
gs = matplotlib.gridspec.GridSpec(3,2)
axs = []
ymax = 0
preTime = 1.5
postTime = 3
maxSpeed = 1000
runPlotTime = np.arange(-preTime,postTime+1/obj.frameRate,1/obj.frameRate)
for stim in ('vis1','vis2','sound1','sound2','catch',None):
    if stim is None:
        i = 2
        j = 1
    elif stim=='catch':
        i = 2
        j = 0
    else:
        i = 0 if '1' in stim else 1
        j = 0 if 'vis' in stim else 1
    ax = fig.add_subplot(gs[i,j])
    axs.append(ax)
    for blockRew,clr in zip(('vis','sound'),'gm'):
        speed = []
        if stim is not None:
            for exps in expsByMouse:
                s = []
                for obj in exps:
                    stimTrials = (obj.trialStim==stim) & (~obj.autoRewarded)
                    blockTrials = np.array([blockRew in s for s in obj.rewardedStim])
                    for st in obj.stimStartTimes[stimTrials & blockTrials]:
                        if st >= preTime and st+postTime <= obj.frameTimes[-1]:
                            ind = (obj.frameTimes >= st-preTime) & (obj.frameTimes <= st+postTime)
                            s.append(np.interp(runPlotTime,obj.frameTimes[ind]-st,obj.runningSpeed[ind]))
                            s[-1][(s[-1]>maxSpeed) | (s[-1]<-maxSpeed)] = np.nan
                speed.append(np.nanmean(s,axis=0))
        if len(speed) > 0:
            m = np.nanmean(speed,axis=0)
            s = np.nanstd(speed,axis=0)/(len(speed)**0.5)
        else:
            m = s = np.full(runPlotTime.size,np.nan)
        ax.plot(runPlotTime,m,color=clr,label=blockRew+' rewarded')
        ax.fill_between(runPlotTime,m+s,m-s,color=clr,alpha=0.25)
        ymax = max(ymax,np.nanmax(m+s))
    if stim is None:
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='center')
    else:
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0,20])
        if i==2 and j==0:
            ax.set_xlabel('Time from stimulus onset (s)')
        if i==1 and j==0:
            ax.set_ylabel('Running speed (cm/s)')
        ax.set_title(stim)
for ax in axs:
    ax.set_ylim([0,1.05*ymax])
plt.tight_layout()

visSpeed = []
soundSpeed = []
for rewStim,speed in zip(('vis1','sound1'),(visSpeed,soundSpeed)):
    for exps in expsByMouse:
        for obj in exps:
            speed.append(np.mean([np.nanmean(obj.runningSpeed[sf-obj.quiescentFrames:sf]) for sf in obj.stimStartFrame[obj.rewardedStim==rewStim]]))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = [0,1.05*max(visSpeed+soundSpeed)]
ax.plot(alim,alim,'--',color='0.5')
ax.plot(visSpeed,soundSpeed,'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('run speed, vis rewarded (cm/s)')
ax.set_ylabel('run speed, sound rewarded (cm/s)')
ax.set_title(str(sum(nExps))+' sessions, '+str(nMice)+' mice')
plt.tight_layout()


# catch rate
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','sound rewarded')):
    dp = []
    for exps in expsByMouse:
        d = np.full((len(exps),6),np.nan)
        for i,obj in enumerate(exps):
            j = obj.blockStimRewarded==rewardStim
            d[i,j] = np.array(obj.catchResponseRate)[j]
        dp.append(np.nanmean(d,axis=0))
    m = np.nanmean(dp,axis=0)
    s = np.nanstd(dp,axis=0)/(len(dp)**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,0.05])
ax.set_xlabel('Block')
ax.set_ylabel('Catch trial response rate')
ax.legend(loc='lower right')
ax.set_title(str(nMice)+' mice')
plt.tight_layout()


# quiescent violations
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','sound rewarded')):
    dp = []
    for exps in expsByMouse:
        d = np.full((len(exps),6),np.nan)
        for i,obj in enumerate(exps):
            for blockInd,blockRewardStim in enumerate(obj.blockStimRewarded):
                if blockRewardStim==rewardStim:
                    trials = obj.trialBlock==blockInd+1
                    d[i,blockInd] = np.sum((obj.quiescentViolationFrames > obj.trialStartFrame[trials][0]) & (obj.quiescentViolationFrames < obj.trialEndFrame[trials][-1]))/trials.sum()
        dp.append(np.nanmean(d,axis=0))
    m = np.nanmean(dp,axis=0)
    s = np.nanstd(dp,axis=0)/(len(dp)**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,0.3])
ax.set_xlabel('Block')
ax.set_ylabel('Quiescent violations per trial')
ax.legend(loc='lower right')
plt.tight_layout()


# simple performance plot
for rewardStim,title in zip(('vis1','sound1'),('visual','auditory')):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1,1,1)
    go = []
    nogo = []
    for exps in expsByMouse:
        goRate = []
        nogoRate = []
        for obj in exps[:5]:
            goRate.extend(np.array(obj.hitRate)[obj.blockStimRewarded==rewardStim])
            nogoRate.extend(np.array(obj.falseAlarmOtherModalGo)[obj.blockStimRewarded!=rewardStim])
        go.append(np.nanmean(goRate))
        nogo.append(np.nanmean(nogoRate))
        y = [go[-1],nogo[-1]] if rewardStim=='vis1' else [nogo[-1],go[-1]]
        ax.plot([0,1],y,'o-',color='k',mec='k',mfc='none',ms=8,alpha=0.25)
    goMean = np.mean(go)
    nogoMean = np.mean(nogo)
    goSem = np.std(go)/(len(go)**0.5)
    nogoSem = np.std(nogo)/(len(nogo)**0.5)
    y = [goMean,nogoMean] if rewardStim=='vis1' else [nogoMean,goMean]
    ax.plot([0,1],y,'o-',color='k',mec='k',mfc='k',ms=12,lw=2)  
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks([0,1])
    ax.set_xticklabels(('visual\ntarget','auditory\ntarget'))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-0.25,1.25])
    ax.set_ylim([0,1.02])
    ax.set_ylabel('Response Rate',fontsize=14)
    ax.set_title(title+' rewarded blocks',fontsize=14)
    plt.tight_layout()
    
    
# simple block switch plot
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 5
postTrials = 15
x = np.arange(-preTrials,postTrials+1)    
for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
        y = []
        for i,exps in enumerate(expsByMouse):
            for j,obj in enumerate(exps):
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0 and rewStim==rewardStim:
                        trials = (obj.trialStim==stim) & ~obj.autoRewarded
                        y.append(np.full(preTrials+postTrials+1,np.nan))
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        k = min(preTrials,pre.size)
                        y[-1][preTrials-k:preTrials] = pre[-k:]
                        post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                        k = min(postTrials,post.size)
                        y[-1][preTrials+1:preTrials+1+k] = post[:k]
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(np.arange(-5,20,5))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch (auto-rewards excluded)',fontsize=12)
    ax.set_ylabel('Response Rate',fontsize=14)
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(blockLabel+'\n'+str(nMice)+' mice, '+str(sum(nExps))+' sessions, '+str(len(y))+' blocks',fontsize=14)
    plt.tight_layout()
    
    
# probability of nogo response since last reward
stimType = ('rewarded target','unrewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
resp = {s: [] for s in stimType}
trialsSincePrevReward= copy.deepcopy(resp)
timeSincePrevReward = copy.deepcopy(resp)
trialsSincePrevResp = copy.deepcopy(resp)
timeSincePrevResp = copy.deepcopy(resp)
for exps in expsByMouse:
    for obj in exps:
        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
            otherModalTarget = np.setdiff1d(obj.blockStimRewarded,rewStim)[0]
            blockTrials = (obj.trialBlock==blockInd+1) & ~obj.catchTrials
            rewTrials = np.where(blockTrials & obj.trialRewarded)[0]
            respTrials = np.where(blockTrials & obj.trialResponse)[0]
            for s in stimType:
                if s=='rewarded target':
                    stim = rewStim
                elif s=='unrewarded target':
                    stim = otherModalTarget
                elif s=='non-target (rewarded modality)':
                    stim = rewStim[:-1]+'2'
                else:
                    stim = otherModalTarget[:-1]+'2'
                nogoTrials = np.where(blockTrials & (obj.trialStim==stim))[0]
                prevRewardTrial = rewTrials[np.searchsorted(rewTrials,nogoTrials) - 1]
                trialsSincePrevReward[s].extend(nogoTrials - prevRewardTrial)
                timeSincePrevReward[s].extend(obj.stimStartTimes[nogoTrials] - obj.stimStartTimes[prevRewardTrial])
                prevRespTrial = respTrials[np.searchsorted(respTrials,nogoTrials) - 1]
                trialsSincePrevResp[s].extend(nogoTrials - prevRespTrial)
                timeSincePrevResp[s].extend(obj.stimStartTimes[nogoTrials] - obj.stimStartTimes[prevRespTrial])
                resp[s].extend(obj.trialResponse[nogoTrials])

for d in (resp,trialsSincePrevReward,timeSincePrevReward,trialsSincePrevResp,timeSincePrevResp):
    for s in stimType:
        d[s] = np.array(d[s])


fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot(1,1,1)
x = np.arange(20)
for s,clr,ls in zip(stimType,'gmgm',('-','-','--','--')):
    n = np.zeros(x.size)
    p = np.zeros(x.size)
    ci = np.zeros((x.size,2))
    for i in x:
        j = trialsSincePrevReward[s]==i
        n[i] += j.sum()
        p[i] += resp[s][j].sum()
    p /= n
    ci = np.array([[b/n[i] for b in scipy.stats.binom.interval(0.95,n[i],p[i])] for i in x])
    ax.plot(x,p,color=clr,ls=ls,label=s)
    ax.fill_between(x,ci[:,0],ci[:,1],color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,14])
ax.set_ylim([0,1.01])
ax.set_xlabel('trials since last reward')
ax.set_ylabel('response rate')
ax.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot(2,1,1)
binWidth = 5
x = np.arange(0,120,binWidth)
for s,clr,ls in zip(stimType,'gmgm',('-','-','--','--')):
    n = np.zeros(x.size)
    p = np.zeros(x.size)
    ci = np.zeros((x.size,2))
    for i,t in enumerate(x):
        j = (timeSincePrevReward[s] > t) & (timeSincePrevReward[s] < t+5)
        n[i] += j.sum()
        p[i] += resp[s][j].sum()
    p /= n
    ci = np.array([[b/n[i] for b in scipy.stats.binom.interval(0.95,n[i],p[i])] for i in range(x.size)])
    ax.plot(x+binWidth/2,p,color=clr,ls=ls,label=s)
    ax.fill_between(x+binWidth/2,ci[:,0],ci[:,1],color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,87.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('time since last reward (s)')
ax.set_ylabel('response rate')
ax.legend(bbox_to_anchor=(1,1),loc='upper left')

ax = fig.add_subplot(2,1,2)
tSort = np.sort(np.concatenate([timeSincePrevReward[s] for s in stimType]))
cumProb = [np.sum(tSort<=i)/tSort.size for i in tSort]
ax.plot(tSort,cumProb,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,87.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('time since last reward (s)')
ax.set_ylabel('cum. prob.')
plt.tight_layout()


    
# transition analysis
blockData = []
for m,exps in enumerate(expsByMouse):
    for obj in exps:#[passSession[m]:]: # exps[:5]:
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            d = {'mouseId':obj.subjectName,
                 'sessionStartTime': obj.startTime,
                 'blockNum':blockInd+1,
                 'goStim':goStim,
                 'numAutoRewards':obj.autoRewarded[:10].sum()}
            blockTrials = obj.trialBlock == blockInd + 1
            firstBlockTrial = np.where(blockTrials)[0][0]
            blockTrials[firstBlockTrial:firstBlockTrial+obj.newBlockNogoTrials] = False
            for trials,lbl in zip((obj.goTrials,obj.sameModalNogoTrials,obj.otherModalGoTrials,obj.otherModalNogoTrials),
                                  ('goTrials','sameModalNogoTrials','otherModalGoTrials','otherModalNogoTrials')):
                trials = trials & blockTrials
                d[lbl] = {'startTimes':obj.stimStartTimes[trials]-obj.blockFirstStimTimes[blockInd],
                          'response':obj.trialResponse[trials],
                          'responseTime':obj.responseTimes[trials]}
            blockData.append(d)

clust = 'all'     
for blockType in ('visual','auditory'):
    goStim = 'vis' if blockType=='visual' else 'sound'
    mid = set()
    session = set()
    nTransitions = 0    
    goProb = []
    goProbPrev = []
    goProbFirst = []
    nogoProb = []
    nogoProbPrev = []
    nogoProbFirst = []
    otherGoProb = []
    otherGoProbPrev = []
    otherGoProbFirst = []
    otherNogoProb = []
    otherNogoProbPrev = []
    otherNogoProbFirst = []
    goLat = []
    goLatPrev = []
    goLatFirst = []
    nogoLat = []
    nogoLatPrev = [] 
    nogoLatFirst = []
    otherGoLat = []
    otherGoLatPrev = []
    otherGoLatFirst = []
    otherNogoLat = []
    otherNogoLatPrev = []
    otherNogoLatFirst = []
    for i,block in enumerate(blockData):
        if clust != 'all' and clustId[i] != clust:
            continue
        if goStim in block['goStim']:
            if block['blockNum'] > 1:
                mid.add(block['mouseId'])
                session.add(block['sessionStartTime'])
                nTransitions += 1
                prevBlock = blockData[i-1]
                goProb.append(block['goTrials']['response'])
                goProbPrev.append(prevBlock['otherModalGoTrials']['response'])
                nogoProb.append(block['sameModalNogoTrials']['response'])
                nogoProbPrev.append(prevBlock['otherModalNogoTrials']['response'])
                otherGoProb.append(block['otherModalGoTrials']['response'])
                otherGoProbPrev.append(prevBlock['goTrials']['response'])
                otherNogoProb.append(block['otherModalNogoTrials']['response'])
                otherNogoProbPrev.append(prevBlock['sameModalNogoTrials']['response'])
                
                goLat.append(block['goTrials']['responseTime'])
                goLatPrev.append(prevBlock['otherModalGoTrials']['responseTime'])
                nogoLat.append(block['sameModalNogoTrials']['responseTime'])
                nogoLatPrev.append(prevBlock['otherModalNogoTrials']['responseTime'])
                otherGoLat.append(block['otherModalGoTrials']['responseTime'])
                otherGoLatPrev.append(prevBlock['goTrials']['responseTime'])
                otherNogoLat.append(block['otherModalNogoTrials']['responseTime'])
                otherNogoLatPrev.append(prevBlock['sameModalNogoTrials']['responseTime'])
            else:
                goProbFirst.append(block['goTrials']['response'])
                nogoProbFirst.append(block['sameModalNogoTrials']['response'])
                otherGoProbFirst.append(block['otherModalGoTrials']['response'])
                otherNogoProbFirst.append(block['otherModalNogoTrials']['response'])
                
                goLatFirst.append(block['goTrials']['responseTime'])
                nogoLatFirst.append(block['sameModalNogoTrials']['responseTime'])
                otherGoLatFirst.append(block['otherModalGoTrials']['responseTime'])
                otherNogoLatFirst.append(block['otherModalNogoTrials']['responseTime'])
    
    colors,labels = ('gm',('visual','auditory')) if blockType=='visual' else ('mg',('auditory','visual'))
    
    preTrials = postTrials = 15 # 15, 45
    x = np.arange(-preTrials,postTrials+1)
    xlim =[-preTrials,postTrials]
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0,1.01]
    ax.plot([0,0],ylim,'k--')
    for first,clr,modal in zip(((goProbFirst,nogoProbFirst),(otherGoProbFirst,otherNogoProbFirst)),colors,labels):
        for r,ls,stim in zip(first,('-','--'),('go','nogo')):
            d = np.full((len(r),preTrials+postTrials+1),np.nan)
            for i,a in enumerate(r):
                j = min(postTrials,a.size)
                d[i,preTrials+1:preTrials+1+j] = a[:j]
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding block switch cue trials)')
    ax.set_ylabel('Response Probability')
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(blockType+' rewarded first block\n('+str(len(goProbFirst)) + ' sessions, ' + str(len(mid))+' mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0.3,0.6]
    ax.plot([0,0],ylim,'k--')
    for first,clr,modal in zip(((goLatFirst,nogoLatFirst),(otherGoLatFirst,otherNogoLatFirst)),colors,labels):
        for r,ls,stim in zip(first,('-','--'),('go','nogo')):
            if 'nogo' in stim:
                continue
            d = np.full((len(r),preTrials+postTrials+1),np.nan)
            for i,a in enumerate(r):
                j = min(postTrials,a.size)
                d[i,preTrials+1:preTrials+1+j] = a[:j]
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding block switch cue trials)')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(loc='upper right')
    ax.set_title(blockType+' rewarded first block\n('+str(len(goLatFirst)) + ' sessions, ' + str(len(mid))+' mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0,1.01]
    ax.plot([0,0],ylim,'k--')
    for a,clr,modal in zip((((goProb,goProbPrev),(nogoProb,nogoProbPrev)),((otherGoProb,otherGoProbPrev),(otherNogoProb,otherNogoProbPrev))),colors,labels):
        for b,ls,stim in zip(a,('-','--'),('go','nogo')):
            current,prev = b
            d = np.full((nTransitions,preTrials+postTrials+1),np.nan)
            for i,r in enumerate(prev):
                j = len(r) if len(r)<preTrials else preTrials
                d[i,preTrials-j:preTrials] = r[-j:] 
            for i,r in enumerate(current):
                j = len(r) if len(r)<postTrials else postTrials
                d[i,preTrials+1:preTrials+1+j] = r[:j] 
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding block switch cue trials)')
    ax.set_ylabel('Response Probability')
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title('transitions to '+blockType+' rewarded blocks\n('+str(nTransitions) +' transitions, ' + str(len(session)) + ' sessions, ' + str(len(mid))+' mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0.3,0.6]
    ax.plot([0,0],ylim,'k--')
    for a,clr,modal in zip((((goLat,goLatPrev),(nogoLat,nogoLatPrev)),((otherGoLat,otherGoLatPrev),(otherNogoLat,otherNogoLatPrev))),colors,labels):
        for b,ls,stim in zip(a,('-','--'),('go','nogo')):
            if 'nogo' in stim:
                continue
            current,prev = b
            d = np.full((nTransitions,preTrials+postTrials+1),np.nan)
            for i,r in enumerate(prev):
                j = len(r) if len(r)<preTrials else preTrials
                d[i,preTrials-j:preTrials] = r[-j:] 
            for i,r in enumerate(current):
                j = len(r) if len(r)<postTrials else postTrials
                d[i,preTrials+1:preTrials+1+j] = r[:j] 
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding block switch cue trials)')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(loc='upper right')
    ax.set_title('transitions to '+blockType+' rewarded blocks\n('+str(nTransitions) +' transitions, ' + str(len(session)) + ' sessions, ' + str(len(mid))+' mice)')
    plt.tight_layout()
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xmax = 9
for goStim,clr,lbl in zip(('vis1','sound1'),'mg',('auditory','visual')):
    trialsToNogo = []
    firstNogoResp = []
    for block in blockData:
        if block['goStim']==goStim:
            trialsToNogo.append(np.sum(block['goTrials']['startTimes'] < block['otherModalGoTrials']['startTimes'][0]) + block['numAutoRewards'])
            firstNogoResp.append(block['otherModalGoTrials']['response'][0])
    trialsToNogo = np.array(trialsToNogo)
    firstNogoResp = np.array(firstNogoResp)
    x = np.unique(trialsToNogo)
    r = [firstNogoResp[trialsToNogo==n] for n in x]
    n = [len(d) for d in r]
    m = np.array([np.nanmean(d) for d in r])
    ci = [np.percentile([np.nanmean(np.random.choice(d,len(d),replace=True)) for _ in range(5000)],(2.5,97.5)) for d in r]
    ax.plot(x[x<=xmax],m[x<=xmax],color=clr,label=lbl)
    for i,c in enumerate(ci):
        if i<=xmax:
            ax.plot([x[i]]*2,c,clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.5,xmax+0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Number of preceeding go stim trials, including autorewards')
ax.set_ylabel('Probability of responding\nto first previous go stim trial')
ax.legend()
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for blockType,goStim,clr in zip(('visual','auditory'),(('vis1','sound1'),('sound1','vis1')),'gm'):
    for stim,trials,ls,lbl in zip(goStim,('goTrials','otherModalGoTrials'),('-',':'),('rewarded','unrewarded')):
        d = [block[trials]['responseTime'] for block in blockData if block['goStim']==stim]
        d = np.concatenate(d)
        d = d[~np.isnan(d)]
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,ls=ls,label=blockType+' '+lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0.1,1])
ax.set_ylim([0,1.01])
ax.set_xlabel('Response Latency (s)')
ax.set_ylabel('Cumulative Probability')
ax.legend(loc='lower right')
plt.tight_layout()
    
    
# plot d prime and response time by block
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded','sound rewarded')):
    dp = []
    for exps in expsByMouse:
        d = np.full((len(exps),6),np.nan)
        for i,obj in enumerate(exps):
            j = obj.blockStimRewarded==rewardStim
            d[i,j] = np.array(obj.dprimeOtherModalGo)[j]
        dp.append(np.nanmean(d,axis=0))
    m = np.nanmean(dp,axis=0)
    s = np.nanstd(dp,axis=0)/(len(dp)**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_yticks(np.arange(0,5,0.5))
ax.set_ylim([0,2.5])
ax.set_xlabel('Block')
ax.set_ylabel('d\' other modality')
ax.legend(loc='lower right')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6)+1
for stim,clr in zip(('vis1','sound1'),'gm'):
    rewStim = ('vis1','sound1') if stim=='vis1' else ('sound1','vis1')
    for rs,ls,alpha,lbl in zip(rewStim,('-','--'),(0.4,0.2),('rewarded','unrewarded')):
        rt = []
        for exps in expsByMouse:
            r = np.full((len(exps),6),np.nan)
            for i,obj in enumerate(exps):
                for j,goStim in enumerate(obj.blockStimRewarded):
                    if rs==goStim:
                        trials = (obj.trialBlock==j+1) & (obj.trialStim==stim) & ~obj.autoRewarded
                        r[i,j] = np.nanmean(obj.responseTimes[trials])
            rt.append(np.nanmean(r,axis=0))
        m = np.nanmean(rt,axis=0)
        s = np.nanstd(rt,axis=0)/(len(rt)**0.5)
        ax.plot(x,m,color=clr,ls=ls,label=stim+' '+lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=alpha)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.3,0.6])
ax.set_xlabel('Block')
ax.set_ylabel('Response time (s)')
ax.legend(loc='upper right')
plt.tight_layout()


# plot mouse learning curve
for ylbl in ('d\' same modality','d\' other modality'):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1,1,1)
    dp = np.full((nMice,max(nExps)),np.nan)
    for i,(exps,clr) in enumerate(zip(expsByMouse,plt.cm.tab20(np.linspace(0,1,nMice)))):
        if 'same' in ylbl:
            d = [np.mean(obj.dprimeSameModal) for obj in exps]
        else:
            d = [np.mean(obj.dprimeOtherModalGo) for obj in exps]
        ax.plot(np.arange(len(d))+1,d,color=clr,alpha=0.25,zorder=2)
        ax.plot(passSession[i]+1,d[passSession[i]],'o',ms=12,color=clr,alpha=0.5,zorder=0)
        dp[i,:len(d)] = d
    m = np.nanmean(dp,axis=0)
    ax.plot(np.arange(len(m))+1,m,'k',lw=2,zorder=1)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    # ax.set_xlim([0,len(m)+1])
    ax.set_xlim([0,36])
    ax.set_ylim([0,2.5])
    ax.set_xlabel('Session',fontsize=14)
    # ax.set_ylabel(ylbl,fontsize=14)
    ax.set_ylabel('d\'',fontsize=14)
    plt.tight_layout()
 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
switchResp = np.full((nMice,max(nExps)),np.nan)
for i,(exps,clr) in enumerate(zip(expsByMouse,plt.cm.tab20(np.linspace(0,1,nMice)))):
    r = []
    for obj in exps:
        rr = []
        for blockInd,goStim in enumerate(obj.blockStimRewarded):
            nogoStim = 'sound1' if goStim=='vis1' else 'vis1'
            rr.append(obj.trialResponse[(obj.trialBlock==blockInd+1) & (obj.trialStim==nogoStim)][0])
        r.append(np.mean(rr))
    ax.plot(np.arange(len(r))+1,r,color=clr,alpha=0.25)
    ax.plot(passSession[i]+1,r[passSession[i]],'o',ms=10,color=clr,alpha=0.25)
    switchResp[i,:len(r)] = r
m = np.nanmean(switchResp,axis=0)
ax.plot(np.arange(len(m))+1,m,'k',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,len(m)+1])
ax.set_ylim([0,1])
ax.set_xlabel('Session')
ax.set_ylabel('First nogo trial resp rate')
plt.tight_layout()
    

# cluster behavior data

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
        
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        k = np.arange(linkageMat.shape[0])+2
        if nreps>0:
            randLinkage = np.zeros((nreps,linkageMat.shape[0]))
            shuffledData = data.copy()
            for i in range(nreps):
                for j in range(data.shape[1]):
                    shuffledData[:,j] = data[np.random.permutation(data.shape[0]),j]
                _,m = cluster(shuffledData,method=method,metric=metric)
                randLinkage[i] = m[::-1,2]
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


# cluster response rate data
stimNames = ('vis1','vis2','sound1','sound2')
clustData = {key: [] for key in ('mouse','session','block','rewardStim','clustData')}
clustData['response'] = {stim: [] for stim in stimNames}
clustData['smoothedResponse'] = {stim: [] for stim in stimNames}
clustData['responseTime'] = {stim: [] for stim in stimNames}
smoothSigma = 5
tintp = np.arange(600)
for m,exps in enumerate(expsByMouse):
    for i,obj in enumerate(exps[passSession[m]:]):
        for blockInd,rewardStim in enumerate(obj.blockStimRewarded):
            clustData['mouse'].append(m)
            clustData['session'].append(i)
            clustData['block'].append(blockInd)
            clustData['rewardStim'].append(rewardStim)
            blockTrials = obj.trialBlock==blockInd+1
            for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                trials = blockTrials & (obj.trialStim==stim) & ~obj.autoRewarded
                stimTime = obj.stimStartTimes[trials]
                stimTime = stimTime-obj.trialStartTimes[trials][0]
                
                clustData['response'][stim].append(obj.trialResponse[trials])
                r = scipy.ndimage.gaussian_filter(obj.trialResponse[trials].astype(float),smoothSigma)
                r = np.interp(tintp,stimTime,r)
                clustData['smoothedResponse'][stim].append(r)
                
                rtTrials = obj.trialResponse[trials]
                if np.any(rtTrials):
                    r = scipy.ndimage.gaussian_filter(obj.responseTimes[trials][rtTrials].astype(float),smoothSigma)
                    r = np.interp(tintp,stimTime[rtTrials],r)
                    clustData['responseTime'][stim].append(r)
                else:
                    clustData['responseTime'][stim].append(np.full(tintp.size,np.nan))
                   
            sn = stimNames if rewardStim=='vis1' else stimNames[-2:]+stimNames[:2]
            clustData['clustData'].append(np.concatenate([clustData['smoothedResponse'][stim][-1] for stim in sn]))

for key in clustData:
    if isinstance(clustData[key],dict):
        for k in clustData[key]:                
            clustData[key][k] = np.array(clustData[key][k])
    else:
        clustData[key] = np.array(clustData[key])


pcaData,eigVal,eigVec = pca(clustData['clustData'],plot=False)
nPC = np.where((np.cumsum(eigVal)/eigVal.sum())>0.95)[0][0]
    

clustColors = [clr for clr in 'krbgmcy']+['0.6']

nClust = 7
    
clustId,linkageMat = cluster(pcaData[:,:nPC],nClusters=nClust,plot=True,colors=clustColors,labels='off',nreps=0)

clustLabels = np.unique(clustId)

minClust = 2
maxClust = 10
clustScores = np.zeros((3,maxClust-minClust+1))
for i,n in enumerate(range(minClust,maxClust+1)):
    cid = cluster(clustData,nClusters=n,plot=False)[0]
    clustScores[0,i] = sklearn.metrics.silhouette_score(pcaData[:,:nPC],cid)
    clustScores[1,i] = sklearn.metrics.calinski_harabasz_score(pcaData[:,:nPC],cid)
    clustScores[2,i] = sklearn.metrics.davies_bouldin_score(pcaData[:,:nPC],cid)


for resp in ('smoothedResponse',):
    for clust in clustLabels:
        for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
                if '1' in stim or resp!='responseTime':
                    r = clustData[resp][stim][(clustData['rewardStim']==rewardStim) & (clustId==clust)]
                    m = np.nanmean(r,axis=0)
                    s = np.nanstd(r)/(len(r)**0.5)
                    ax.plot(tintp,m,color=clr,lw=2,ls=ls,label=stim)
                    ax.fill_between(tintp,m+s,m-s,color=clr,alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=14)
            ax.set_ylim([0,1.02])
            ax.set_xlabel('Time (s)',fontsize=16)
            ax.set_ylabel('Response rate',fontsize=16)
            ax.legend(loc='lower right',fontsize=14)
            ax.set_title('Cluster '+str(clust)+', '+blockLabel+' (n='+str(len(r))+')',fontsize=18)
            plt.tight_layout()


postTrials = 15
x = np.arange(postTrials)+1
for clust in clustLabels:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for stim,clr,ls in zip(stimNames,'ggmm',('-','--','-','--')):
            resp = []
            for r in clustData['response'][stim][(clustData['rewardStim']==rewardStim) & (clustId==clust)]:
                j = min(postTrials,r.size)
                resp.append(np.full(postTrials,np.nan))
                resp[-1][:j] = r[:j]
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
        ax.set_ylabel('Response rate')
        ax.legend(loc='lower right')
        ax.set_title('Cluster '+str(clust)+', '+blockLabel+' (n='+str(len(resp))+')')
        plt.tight_layout()
        


fig = plt.figure()
ax = fig.add_subplot(1,1,1)   
for rewardStim,clr,lbl in zip(('vis1','sound1'),'gm',('visual rewarded blocks','sound rewarded blocks')):
    y = []
    for clust in clustLabels:
        blocks = d['rewardStim']==rewardStim
        y.append(np.sum(blocks & (clustId==clust))/blocks.sum())
    ax.plot(clustLabels,y,color=clr,lw=2,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xticks(clustLabels)
ax.set_ylim([0,0.5])
ax.set_xlabel('Cluster',fontsize=16)
ax.set_ylabel('Probability',fontsize=16)
ax.legend(fontsize=14)
plt.tight_layout()


mouseClustProb = np.zeros((nMice,6))
ind = 0
for i,n in enumerate(nExps):
    for j,clust in enumerate(clustLabels):
        mouseClustProb[i,j] = np.sum(clustId[ind:ind+n]==clust)/n
    ind += n

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
im = ax.imshow(mouseClustProb,cmap='magma',clim=(0,mouseClustProb.max()))
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(6))
ax.set_xticklabels(np.arange(6)+1)
yticks = np.concatenate(([0],np.arange(4,nMice+1,5)))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks+1)
ax.set_xlabel('Cluster')
ax.set_ylabel('Mouse')
ax.set_title('Probability')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)   
ax.bar(clustLabels,np.sum(mouseClustProb>0,axis=0)/nMice,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(clustLabels)
ax.set_ylim([0,1.01])
ax.set_xlabel('Cluster',fontsize=14)
ax.set_ylabel('Fraction of mice contributing to cluster',fontsize=14)
plt.tight_layout()


sessionClustProb = np.zeros((sum(nExps),6))
ind = 0
for i in range(sum(nExps)):
    for j,clust in enumerate(clustLabels):
        sessionClustProb[i,j] = np.sum(clustId[ind:ind+6]==clust)/6
    ind += 6

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
im = ax.imshow(sessionClustProb,cmap='magma',clim=(0,sessionClustProb.max()),aspect='auto',interpolation='none')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(6))
ax.set_xticklabels(np.arange(6)+1)
# yticks = np.concatenate(([0],np.arange(4,nMice+1,5)))
# ax.set_yticks(yticks)
# ax.set_yticklabels(yticks+1)
ax.set_xlabel('Cluster')
ax.set_ylabel('Session')
ax.set_title('Probability')
plt.tight_layout()


blockClustProb = np.zeros((6,len(clustLabels)))
for i in range(6):
    blocks = d['block']==i
    for j,clust in enumerate(clustLabels):
        blockClustProb[i,j] = np.sum(blocks & (clustId==clust))/blocks.sum()

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
im = ax.imshow(blockClustProb,cmap='magma',clim=(0,blockClustProb.max()),origin='lower')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(len(clustLabels)))
ax.set_xticklabels(np.arange(6)+1)
ax.set_yticklabels(clustLabels)
ax.set_xlabel('Cluster')
ax.set_ylabel('Block')
ax.set_title('Probability')
plt.tight_layout()

chanceProb = np.array([np.sum(clustId==clust)/len(clustId) for clust in clustLabels])

for lbl in ('Absolute','Relative'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    a = blockClustProb-chanceProb
    if lbl=='Relative':
        a /= chanceProb
    amax = np.absolute(a).max()
    im = ax.imshow(a,clim=(-amax,amax),cmap='bwr',origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(len(clustLabels)))
    ax.set_xticklabels(np.arange(6)+1)
    ax.set_yticklabels(clustLabels)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Block')
    ax.set_title(lbl+' difference from chance probability')
    plt.tight_layout()

nIter = int(1e5)
randClust = np.stack([np.random.choice(clustLabels,len(clustId),replace=True,p=chanceProb) for _ in range(nIter)])
randClustProb = np.array([[np.sum(r==clust)/len(clustId) for clust in clustLabels] for r in randClust])

pval = np.zeros_like(blockClustProb)
for i,p in enumerate(blockClustProb.T):
    lessThan = np.sum(randClustProb<p,axis=0)/randClustProb.shape[0]
    greaterThan = np.sum(randClustProb>p,axis=0)/randClustProb.shape[0]
    pval[i] = np.min(np.stack((lessThan,greaterThan)),axis=0)
pval[pval==0] = 1/nIter

alpha = 0.05
pvalCorr = np.reshape(multipletests(pval.flatten(),alpha=alpha,method='fdr_bh')[1],pval.shape)

fig = plt.figure(facecolor='w')
ax = fig.subplots(1)
lim = (10**np.floor(np.log10(np.min(pvalCorr))),alpha)
clim = np.log10(lim)
im = ax.imshow(np.log10(pvalCorr),cmap='gray',clim=clim,origin='lower')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
cb.ax.tick_params(labelsize=10) 
legticks = np.concatenate((np.arange(clim[0],clim[-1]),[clim[-1]]))
cb.set_ticks(legticks)
cb.set_ticklabels(['$10^{'+str(int(lt))+'}$' for lt in legticks[:-1]]+[r'$\geq0.05$'])
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(len(clustLabels)))
ax.set_xticklabels(np.arange(6)+1)
ax.set_yticklabels(clustLabels)
ax.set_xlabel('Cluster')
ax.set_ylabel('Block')
ax.set_title('Corrected p-value')
plt.tight_layout()


prevClustProb = np.zeros((len(clustLabels),)*2)
blocks = np.where(d['block']>0)[0]
for j,clust in enumerate(clustLabels):
    c = clustId[blocks]==clust
    for i,prevClust in enumerate(clustLabels):
        prevClustProb[i,j] = np.sum(clustId[blocks-1][c]==prevClust)/c.sum()

nextClustProb = np.zeros((len(clustLabels),)*2)
blocks = np.where(d['block']<5)[0]
for j,clust in enumerate(clustLabels):
    c = clustId[blocks]==clust
    for i,nextClust in enumerate(clustLabels):
        nextClustProb[i,j] = np.sum(clustId[blocks+1][c]==nextClust)/c.sum()

for transProb,lbl in zip((prevClustProb,nextClustProb),('Previous','Next')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    im = ax.imshow(transProb,cmap='magma',clim=(0,transProb.max()),origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    ax.set_xticks(np.arange(len(clustLabels)))
    ax.set_yticks(np.arange(len(clustLabels)))
    ax.set_xticklabels(clustLabels)
    ax.set_yticklabels(clustLabels)
    ax.set_xlabel('Current block cluster')
    ax.set_ylabel(lbl+' block cluster')
    ax.set_title('Probability')
    plt.tight_layout()

chanceProb = np.array([np.sum(clustId[blocks+1]==clust)/len(blocks) for clust in clustLabels])

for transProb,lbl in zip((prevClustProb,nextClustProb),('Previous','Next')):
    for diff in ('Absolute','Relative'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        a = transProb-chanceProb[:,None]
        if diff=='Relative':
            a /= chanceProb[:,None]
        amax = np.absolute(a).max()
        im = ax.imshow(a,clim=(-amax,amax),cmap='bwr',origin='lower')
        cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
        ax.set_xticks(np.arange(len(clustLabels)))
        ax.set_yticks(np.arange(len(clustLabels)))
        ax.set_xticklabels(clustLabels)
        ax.set_yticklabels(clustLabels)
        ax.set_xlabel('Current block cluster')
        ax.set_ylabel(lbl+' block cluster')
        ax.set_title(diff+' difference from chance probability')
        plt.tight_layout()

nIter = int(1e5)
randClust = np.stack([np.random.choice(clustLabels,len(blocks),replace=True,p=chanceProb) for _ in range(nIter)])
randClustProb = np.array([[np.sum(r==clust)/len(blocks) for clust in clustLabels] for r in randClust])

for transProb,lbl in zip((prevClustProb,nextClustProb),('Previous','Next')):
    pval = np.zeros_like(transProb)
    for j,p in enumerate(transProb.T):
        lessThan = np.sum(randClustProb<p,axis=0)/randClustProb.shape[0]
        greaterThan = np.sum(randClustProb>p,axis=0)/randClustProb.shape[0]
        pval[:,j] = np.min(np.stack((lessThan,greaterThan)),axis=0)
    pval[pval==0] = 1/nIter
    
    alpha = 0.05
    pvalCorr = np.reshape(multipletests(pval.flatten(),alpha=alpha,method='fdr_bh')[1],pval.shape)
    
    fig = plt.figure(facecolor='w')
    ax = fig.subplots(1)
    lim = (10**np.floor(np.log10(np.min(pvalCorr))),alpha)
    clim = np.log10(lim)
    im = ax.imshow(np.log10(pvalCorr),cmap='gray',clim=clim,origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    cb.ax.tick_params(labelsize=10) 
    legticks = np.concatenate((np.arange(clim[0],clim[-1]),[clim[-1]]))
    cb.set_ticks(legticks)
    cb.set_ticklabels(['$10^{'+str(int(lt))+'}$' for lt in legticks[:-1]]+[r'$\geq0.05$'])
    ax.set_xticks(np.arange(len(clustLabels)))
    ax.set_yticks(np.arange(len(clustLabels)))
    ax.set_xticklabels(clustLabels)
    ax.set_yticklabels(clustLabels)
    ax.set_xlabel('Current block cluster')
    ax.set_ylabel(lbl+' block cluster')
    ax.set_title('Corrected p-value')
    plt.tight_layout()


# regression model
nTrialsPrev = 15
stimNames = ('vis1','vis2','sound1','sound2')
regressors = ('reinforcement',
              'crossModalReinforcement',
              'reward','action')
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
regData['sampleWeight'] = []
s = -1
b = -1
for m,exps in enumerate(expsByMouse):
    for sn,obj in enumerate(exps):
        s += 1
        for blockInd in range(6):
            b += 1
            if blockInd==0:
                continue
            trials = ~obj.catchTrials & ~obj.autoRewarded & (obj.trialBlock==blockInd+1)
            trialInd = np.where(trials)[0]
            nTrials = trials.sum()
            regData['X'].append({})
            regData['sampleWeight'].append(np.ones(trials.sum()))
            for stim in stimNames:
                regData['X'][-1][stim] = (obj.trialStim[trials]==stim)[:,None]
            for r in regressors:
                regData['X'][-1][r] = np.zeros((nTrials,nTrialsPrev))
                for n in range(1,nTrialsPrev+1):
                    for trial,stim in enumerate(obj.trialStim[trials]):
                        if stim!=obj.blockStimRewarded[blockInd] and stim in obj.blockStimRewarded:
                            regData['sampleWeight'][-1][trial] *= 10
                        if r in ('reinforcement','noReinforcement','preservation'):
                            sameStim = obj.trialStim[:trialInd[trial]] == stim
                            if sameStim.sum()>n:
                                if r in ('reinforcement','noReinforcement'):
                                    if obj.trialResponse[:trialInd[trial]][sameStim][-n]:
                                        rew = obj.trialRewarded[:trialInd[trial]][sameStim][-n]
                                        # if (r=='reinforcement' and rew) or (r=='noReinforcement' and not rew):
                                        #     X[-1][r][trial,n-1] = 1
                                        regData['X'][-1][r][trial,n-1] = 1 if rew else -1
                                elif r=='preservation':
                                    regData['X'][-1][r][trial,n-1] = obj.trialResponse[:trialInd[trial]][sameStim][-n]
                        else:
                            notCatch = obj.trialStim[:trialInd[trial]] != 'catch'
                            if notCatch.sum()>n:
                                resp = obj.trialResponse[:trialInd[trial]][notCatch][-n]
                                rew = obj.trialRewarded[:trialInd[trial]][notCatch][-n]
                                if r in ('crossModalReinforcement','crossModalNoReinforcement') and resp:
                                    if not any(s in stim and s in obj.trialStim[:trialInd[trial]][notCatch][-n] for s in ('vis','sound')):
                                        # if (r=='crossModalReinforcement' and rew) or (r=='crossModalNoReinforcement' and not rew):
                                        #     X[-1][r][trial,n-1] = 1
                                        regData['X'][-1][r][trial,n-1] = 1 if rew else -1
                                elif r=='reward' and rew:
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


def crossValidate(model,X,y,nSplits,sampleWeight=None):
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
    cv['intercept'] = []
    modelMethods = dir(model)
    
    for k,estimator in enumerate(cv['estimator']):
        testInd = []
        for val,n in zip(classVals,samplesPerSplit):
            start = k*n
            ind = shuffleInd[y[shuffleInd]==val]
            testInd.extend(ind[start:start+n] if k+1<nSplits else ind[start:])
        if len(testInd) > 0:
            trainInd = np.setdiff1d(shuffleInd,testInd) if nSplits > 1 else testInd
            sw = None if sampleWeight is None else sampleWeight[trainInd]
            estimator.fit(X[trainInd],y[trainInd],sample_weight=sw)
            cv['train_score'].append(estimator.score(X[trainInd],y[trainInd]))
            cv['test_score'].append(estimator.score(X[testInd],y[testInd]))
            cv['predict'][testInd] = estimator.predict(X[testInd])
            for method in ('predict_proba','decision_function'):
                if method in modelMethods:
                    cv[method][testInd] = getattr(estimator,method)(X[testInd])
            for attr in ('feature_importance_','coef_','intercept_'):
                if attr in estimator.__dict__:
                    cv[attr[:-1]].append(getattr(estimator,attr))
    return cv


holdOutRegressor = ('none',)+regressors+(('reinforcement','crossModalReinforcement'),('reward','action'))
accuracy = {h: [] for h in holdOutRegressor}
trainAccuracy = copy.deepcopy(accuracy)
balancedAccuracy = copy.deepcopy(accuracy)
prediction = copy.deepcopy(accuracy)
predictProb = copy.deepcopy(accuracy)
confidence = copy.deepcopy(accuracy)
featureWeights = copy.deepcopy(accuracy)
bias = copy.deepcopy(accuracy)
model = LogisticRegression(fit_intercept=True,max_iter=1e3)
for h in holdOutRegressor:
    # fit all data
    # print(h)
    # x = np.concatenate([np.concatenate([b[r] for r in stimNames+regressors if r!=h and r not in h],axis=1) for b in regData['X']])
    # y = np.concatenate(regData['trialResponse'])
    # cv = crossValidate(model,x,y,nSplits=5)
    # accuracy[h].append(np.mean(cv['test_score']))
    # trainAccuracy[h].append(np.mean(cv['train_score']))
    # balancedAccuracy[h].append(sklearn.metrics.balanced_accuracy_score(y,cv['predict']))
    # prediction[h].append(cv['predict'])
    # predictProb[h].append(cv['predict_proba'])
    # confidence[h].append(cv['decision_function'])
    # featureWeights[h].append(np.mean(cv['coef'],axis=0)[0])
    # bias[h].append(np.mean(cv['intercept']))
    
    # fit each session
    for s in np.unique(regData['sessionIndex']):
        print(h,s)
        x = []
        y = []
        for b in range(len(regData['blockIndex'])):
            if regData['sessionIndex'][b]==s:
                x.append(np.concatenate([regData['X'][b][r] for r in regressors if r!=h and r not in h],axis=1))
                y.append(regData['trialResponse'][b])
        x = np.concatenate(x)
        y = np.concatenate(y)
        cv = crossValidate(model,x,y,nSplits=10)
        accuracy[h].append(np.mean(cv['test_score']))
        trainAccuracy[h].append(np.mean(cv['train_score']))
        balancedAccuracy[h].append(sklearn.metrics.balanced_accuracy_score(y,cv['predict']))
        prediction[h].append(cv['predict'])
        predictProb[h].append(cv['predict_proba'])
        confidence[h].append(cv['decision_function'])
        featureWeights[h].append(np.mean(cv['coef'],axis=0)[0])
        bias[h].append(np.mean(cv['intercept']))
    break


regressorColors = 'rgmbyck'[:len(regressors)]
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
    d = np.array(featureWeights[h])[:,4:]
    reg,clrs = zip(*[(r,c) for r,c in zip(regressors,regressorColors) if r!=h and r not in h])
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


postTrials = 15
x = np.arange(postTrials)+1
for h in [('reward','action')]: #holdOutRegressor:
    modelResponse = []
    #if fit all data
    i = 0
    for r in regData['trialResponse']:
        modelResponse.append(prediction[h][0][i:i+len(r)])
        i += len(r)
    
    # if fit each session
    # for s,pred in enumerate(prediction[h]):
    #     i = 0
    #     for r in (r for r,si in zip(regData['trialResponse'],regData['sessionIndex']) if si==s):
    #          modelResponse.append(pred[i:i+len(r)])
    #          i += len(r)
             
    for clust in clustLabels:
        fig = plt.figure()
        f = 1
        for d,ylbl in zip((regData['trialResponse'],modelResponse),('mice','model')):
            for rewStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','sound rewarded blocks')):
                ax = fig.add_subplot(2,2,f)
                for stim,clr,ls,lbl in zip(stimNames,'ggmm',('-','--','-','--'),('visual go','visual nogo','auditory go','auditory nogo')):
                    resp = []
                    for j,r in enumerate(d):
                        if regData['rewardStim'][j]==rewStim and clustId[regData['blockIndex'][j]]==clust:
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


sessions = np.unique(regData['sessionIndex'])
earlySessions = np.zeros(sessions.size,dtype=bool)
lateSessions = np.zeros_like(earlySessions)
mcount = 0
scount = 0
for i,m in enumerate(regData['mouseIndex'][::5]):
    if m > mcount:
        scount = 0
        mcount = m
    if scount<5:
        earlySessions[i] = True
    if passSession[m] <= scount:# < passSession[m]+5:
        lateSessions[i] = True
    scount += 1

x = np.arange(nTrialsPrev)+1
for h in holdOutRegressor:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    d = np.array(featureWeights[h])[lateSessions,:]
    reg,clrs = zip(*[(r,c) for r,c in zip(regressors,regressorColors) if r!=h and r not in h])
    mean = np.median(d,axis=0)
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


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
mcount = 0
scount = 0
r = [np.full(100,np.nan)]
c = [np.full(100,np.nan)]
fw = np.array(featureWeights['none'])[:,:]
for i,m in enumerate(regData['mouseIndex'][::5]):
    if m > mcount:
        scount = 0
        mcount = m
        r.append(np.full(100,np.nan))
        c.append(np.full(100,np.nan))
    r[-1][scount] = fw[i,:15].sum()
    c[-1][scount] = fw[i,15:30].sum()
    scount += 1
ax.plot(np.nanmean(r,axis=0),'r')
ax.plot(np.nanmean(c,axis=0),'b')



# response probability as function of preceeding go or nogo trials/responses
nGo = np.zeros((100,100))
nNogo = np.zeros((100,100))
goResp = np.zeros((100,100))
nogoResp = np.zeros((100,100))
for exps in expsByMouse:
    for obj in exps:
        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock==blockInd+1
            rew = np.concatenate(([0],np.cumsum(blockTrials & obj.trialResponse & (obj.trialStim==obj.rewardedStim))[:-1]))
            noRew = np.concatenate(([0],np.cumsum(blockTrials & obj.trialResponse & (obj.trialStim!=obj.rewardedStim) & np.in1d(obj.trialStim,obj.blockStimRewarded))[:-1]))
            for i in np.where(blockTrials)[0]:
                if obj.trialStim[i]==obj.rewardedStim[i]:
                    nGo[rew[i],noRew[i]] += 1
                    if obj.trialResponse[i]:
                        goResp[rew[i],noRew[i]] += 1
                elif obj.trialStim[i] in obj.blockStimRewarded:
                    nNogo[rew[i],noRew[i]] += 1
                    if obj.trialResponse[i]:
                        nogoResp[rew[i],noRew[i]] += 1

minTrials = 5
for r,n,lbl in zip((goResp,nogoResp),(nGo,nNogo),('go stim','nogo stim')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    a = r/n
    a[n<minTrials] = np.nan
    im = ax.imshow(a,clim=(0,1),cmap='magma',origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    ax.set_xlim([-0.5,20.5])
    ax.set_ylim([-0.5,20.5])
    ax.set_xlabel('# non-rewarded responses (nogo stim)')
    ax.set_ylabel('# rewarded responses (go stim)')
    ax.set_title('Response rate ('+lbl+')')
    plt.tight_layout()
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
r = np.zeros(100)
n = np.zeros(100)    
for exps in expsByMouse:
    for obj in exps:
        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock==blockInd+1
            goTrials = obj.trialStim[blockTrials]==rewStim
            nogoTrials = np.in1d(obj.trialStim[blockTrials],obj.blockStimRewarded) & ~goTrials
            firstNogo = np.where(nogoTrials)[0][0]
            i = np.sum(goTrials[:firstNogo])
            r[i] += obj.trialResponse[blockTrials][firstNogo]
            n[i] += 1
p = r/n
p[n<10] = np.nan
p[:5] = np.nan
ax.plot(np.arange(p.size),p,'ko',ms=10)
for x in np.where(~np.isnan(p))[0]:
    s = [b/n[x] for b in scipy.stats.binom.interval(0.95,n[x],p[x])]
    ax.plot([x,x],s,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.5,10.5])
ax.set_ylim([0,1])
ax.set_xlabel('# rewarded responses (go stim)')
ax.set_ylabel('Response rate, first nogo stim')
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
r = np.zeros(100)
n = np.zeros(100)    
for exps in expsByMouse:
    for obj in exps:
        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock==blockInd+1
            goTrials = obj.trialStim[blockTrials]==rewStim
            nogoTrials = np.in1d(obj.trialStim[blockTrials],obj.blockStimRewarded) & ~goTrials
            firstGo = np.where(goTrials)[0][0]
            i = np.sum((nogoTrials & obj.trialResponse[blockTrials])[:firstGo])
            r[i] += obj.trialResponse[blockTrials][firstGo]
            n[i] += 1
p = r/n
p[n<5] = np.nan
ax.plot(np.arange(p.size),p,'ko',ms=10)
for x in np.where(~np.isnan(p))[0]:
    s = [b/n[x] for b in scipy.stats.binom.interval(0.95,n[x],p[x])]
    ax.plot([x,x],s,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.5,10.5])
ax.set_ylim([0,1])
ax.set_xlabel('# non-rewarded responses (nogo stim)')
ax.set_ylabel('Response rate, first go stim')
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
r = np.zeros(100)
n = np.zeros(100)    
for exps in expsByMouse:
    for obj in exps:
        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock==blockInd+1
            goTrials = obj.trialStim[blockTrials]==rewStim
            for j,trial in enumerate(np.where(goTrials)[0]):
                i = obj.trialResponse[blockTrials][goTrials][:j].sum()
                r[i] += obj.trialResponse[blockTrials][trial]
                n[i] += 1
p = r/n
p[n<5] = np.nan
ax.plot(np.arange(p.size),p,'ko',ms=10)
for x in np.where(~np.isnan(p))[0]:
    s = [b/n[x] for b in scipy.stats.binom.interval(0.95,n[x],p[x])]
    ax.plot([x,x],s,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.5,10.5])
ax.set_ylim([0,1])
ax.set_xlabel('# rewarded responses (go stim)')
ax.set_ylabel('Response rate, go stim')
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
r = np.zeros(100)
n = np.zeros(100)    
for exps in expsByMouse:
    for obj in exps:
        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
            blockTrials = obj.trialBlock==blockInd+1
            goTrials = obj.trialStim[blockTrials]==rewStim
            nogoTrials = np.in1d(obj.trialStim[blockTrials],obj.blockStimRewarded) & ~goTrials
            for trial in np.where(nogoTrials)[0]:
                ind = goTrials.copy()
                ind[trial:] = False
                i = obj.trialResponse[blockTrials][ind].sum()
                r[i] += obj.trialResponse[blockTrials][trial]
                n[i] += 1
p = r/n
p[n<5] = np.nan
ax.plot(np.arange(p.size),p,'ko',ms=10)
for x in np.where(~np.isnan(p))[0]:
    s = [b/n[x] for b in scipy.stats.binom.interval(0.95,n[x],p[x])]
    ax.plot([x,x],s,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.5,10.5])
ax.set_ylim([0,1])
ax.set_xlabel('# rewarded responses (go stim)')
ax.set_ylabel('Response rate, nogo stim')
plt.tight_layout()












