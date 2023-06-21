# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:35:34 2023

@author: svc_ccg
"""

import copy
import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

hitThresh = 100
dprimeThresh = 1.5

# number of sessions trained after stage 2
hasOldRegimens = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var']
oldMice,newMice = [np.array(summaryDf[ind & summaryDf['stage 2 pass'] & ~summaryDf['wheel fixed']]['mouse id']) for ind in (hasOldRegimens,~hasOldRegimens)]

sessionsToPassOld = []
sessionsToPassNew = []
for mice,sessionsToPass in zip((oldMice,newMice),(sessionsToPassOld,sessionsToPassNew)):
    for mid in mice:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        firstExperimentSession = np.where(['multimodal' in task
                                           or 'contrast'in task
                                           or 'opto' in task
                                           or 'nogo' in task
                                           or 'noAR' in task
                                           #or 'NP' in rig 
                                           for task,rig in zip(df['task version'],df['rig name'])])[0]
        if len(firstExperimentSession)>0:
            sessions[firstExperimentSession[0]:] = False
        sessions = np.where(sessions)[0]
        passed = False
        if len(sessions) > 0:
            for ind in sessions[1:]:
                if isinstance(df.loc[ind,'hits'],str):
                    dprimeSame = [[float(s) for s in re.findall('-*[0-9].[0-9]*',df.loc[i,'d\' same modality'])] for i in (ind,ind-1)]
                    dprimeOther = [[float(s) for s in re.findall('-*[0-9].[0-9]*',df.loc[i,'d\' other modality go stim'])] for i in (ind,ind-1)]
                else:
                    dprimeSame = [df.loc[i,'d\' same modality'] for i in (ind,ind-1)]
                    dprimeOther = [df.loc[i,'d\' other modality go stim'] for i in (ind,ind-1)]
                
                if np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                    sessionsToPass.append(ind - sessions[0] + 1)
                    if mid in oldMice:
                        sessionsToPass[-1] += sessions[0] - (np.where(np.array(['stage 2' in task for task in df['task version']]))[0][-1] + 1)
                    passed = True
                    break
        if not passed:
            sessionsToPass.append(np.nan)
        


# old
mouseIds = df['mouse id']
passOnly = True
lateAutoRewardOnly = False

mice = []
sessionStartTimes = []
passSession =[]
movingGrating = []
amNoise = []
for mid in mouseIds:
    if str(mid) in sheets:
        mouseInd = np.where(allMiceDf['mouse id']==int(mid))[0][0]
        df = sheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        # sessions = np.array(['stage 5' in task and 'tone' in task and 'moving' not in task for task in df['task version']])
        # sessions = np.array(['stage 5' in task and 'tone' in task and 'moving' in task for task in df['task version']])
        # sessions = np.array(['stage 5' in task and 'AMN' in task and 'moving' not in task for task in df['task version']])
        # sessions = np.array(['stage 5' in task and 'AMN' in task and 'moving' in task for task in df['task version']])
        if sessions.sum() == 0:
            continue
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
        if not lateAutoRewardOnly or (obj.autoRewardOnsetFrame >= obj.responseWindow[1]):
            expsByMouse[-1].append(obj)
        
nMice = len(expsByMouse)
nExps = [len(exps) for exps in expsByMouse]


    
# block switch plot, all stimuli
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)    
for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
        y = []
        for exps in expsByMouse:
            y.append(np.full((len(exps),preTrials+postTrials+1),np.nan))
            for i,obj in enumerate(exps):
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0 and rewStim==rewardStim:
                        trials = (obj.trialStim==stim) & ~obj.autoRewarded 
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        j = min(preTrials,pre.size)
                        y[-1][i][preTrials-j:preTrials] = pre[-j:]
                        post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                        j = min(postTrials,post.size)
                        y[-1][i][preTrials+1:preTrials+1+j] = post[:j]
            y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks(np.arange(-20,20,5))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch (auto-rewards excluded)',fontsize=12)
    ax.set_ylabel('Response Rate',fontsize=14)
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(blockLabel+'\n'+str(nMice)+' mice',fontsize=14)
    plt.tight_layout()
    
    
# probability of nogo response since last reward
stimType = ('rewarded target','unrewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
resp = {s: [] for s in stimType}
trialsSincePrevReward= copy.deepcopy(resp)
trialsSincePrevResp = copy.deepcopy(resp)
trialsSincePrevStim = copy.deepcopy(resp)
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
                stimTrials = np.where(blockTrials & (obj.trialStim==stim))[0]
                prevRewardTrial = rewTrials[np.searchsorted(rewTrials,stimTrials) - 1]
                prevRespTrial = respTrials[np.searchsorted(respTrials,stimTrials) - 1]
                trialsSincePrevReward[s].extend(stimTrials - prevRewardTrial)
                trialsSincePrevResp[s].extend(stimTrials - prevRespTrial)
                trialsSincePrevStim[s].extend(np.concatenate(([np.nan],np.diff(stimTrials))))
                resp[s].extend(obj.trialResponse[stimTrials])

for d in (trialsSincePrevReward,trialsSincePrevResp,trialsSincePrevStim,resp):
    for s in stimType:
        d[s] = np.array(d[s])


for trialsSince,lbl in zip((trialsSincePrevReward,trialsSincePrevResp,trialsSincePrevStim),
                           ('reward','response','same stimulus')):
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(20)
    for s,clr,ls in zip(stimType,'gmgm',('-','-','--','--')):
        n = np.zeros(x.size)
        p = np.zeros(x.size)
        ci = np.zeros((x.size,2))
        for i in x:
            j = trialsSince[s]==i
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
    ax.set_xlabel('trials since last '+lbl)
    ax.set_ylabel('response rate')
    ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()



































