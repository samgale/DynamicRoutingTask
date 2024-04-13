# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:05:04 2023

@author: svc_ccg
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import npc_lims
import npc_sessions


sessionInfo = npc_lims.get_session_info()

projectNames = list(set(s.project for s in sessionInfo))

mouseIds = np.array([s.subject for s in sessionInfo])


# find mice with ephys sessions
ephysMice = []
for s in sessionInfo:
    if s.project in ('DynamicRouting','DRPilotSession') and s.is_ephys:
        if s.subject not in ephysMice:
            ephysMice.append(s.subject)

            
# build sessions table
colNames = ('id','mouse','date','rig','ephys')
sessionDict = {col: [] for col in colNames}
for s in sessionInfo:
    sid = s.id.id
    if s.subject in ephysMice:
        session = npc_sessions.DynamicRoutingSession(sid,is_sync=False)
        sessionDict['id'].append(sid)
        sessionDict['mouse'].append(s.subject)
        sessionDict['date'].append(s.date)
        sessionDict['rig'].append(session.rig)
        sessionDict['ephys'].append(s.is_ephys)
sessionDf = pd.DataFrame(sessionDict)


# get trials table for each session
preHabSessions = []
habSessions = []
ephysSessions = []
badSessions = []
for mouse in ephysMice:
    df = sessionDf[sessionDf['mouse']==mouse].sort_values('date').reset_index()
    rig = np.array(df['rig']=='NP3')
    ephys = np.array(df['ephys'])
    hab = rig & ~ephys
    if np.any(hab):
        firstHab = np.where(hab)[0][0]
        preHabSessions.append([])
        for sid in df['id'][firstHab-5:firstHab]:
            try:
                d = npc_sessions.DynamicRoutingSession(sid,is_sync=False).trials[:]
                if d['block_index'].max() == 5:
                    preHabSessions[-1].append(d)
            except:
                badSessions.append(sid)
        habSessions.append([])
        for sid in df['id'][hab]:
            try:
                d = npc_sessions.DynamicRoutingSession(sid,is_sync=False).trials[:]
                if d['block_index'].max() == 5:
                    habSessions[-1].append(d)
            except:
                badSessions.append(sid)
    ephysSessions.append([])
    for sid in df['id'][ephys]:
        try:
            d = npc_sessions.DynamicRoutingSession(sid,is_sync=False).trials[:]
            if d['block_index'].max() == 5:
                ephysSessions[-1].append(d)
        except:
            badSessions.append(sid)
    # firstHab = np.where(hab)[0][0]
    # preHabSessions.append([npc_sessions.DynamicRoutingSession(sid,is_sync=False).trials[:] for sid in df['id'][firstHab-5:firstHab]])
    # habSessions.append([npc_sessions.DynamicRoutingSession(sid,is_sync=False).trials[:] for sid in df['id'][hab]])
    # ephysSessions.append([npc_sessions.DynamicRoutingSession(sid,is_sync=False).trials[:] for sid in df['id'][ephys]])



# block switch plot, all stimuli
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)
for sessions,sessionsLbl in zip((preHabSessions,habSessions,ephysSessions),('pre-habituation','habituation','ephys')):
    if sessionsLbl!='ephys':
        continue
    for blockRewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
            y = []
            for sessionTables in sessions: 
                y.append([])
                for d in sessionTables:
                    trialBlock = np.array(d['block_index'])
                    trialResp = np.array(d['is_response'])
                    trialStim = np.array(d['stim_name']) 
                    goStim = np.array(d['is_go'])
                    autoReward = np.array(d['is_reward_scheduled'])
                    for blockInd in np.unique(trialBlock): # range(1,6)
                        rewStim = trialStim[(trialBlock==blockInd) & goStim][0]
                        if blockInd > 0 and rewStim == blockRewardStim:
                            trials = (trialStim==stim) & ~autoReward
                            y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                            pre = trialResp[(trialBlock==blockInd-1) & trials]
                            i = min(preTrials,pre.size)
                            y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                            post = trialResp[(trialBlock==blockInd) & trials]
                            i = min(postTrials,post.size)
                            y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
                y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(-20,20,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch\n(excluding auto-rewards)',fontsize=12)
        ax.set_ylabel('Response Rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        ax.set_title(sessionsLbl+' ('+str(len(y))+' mice)\n'+blockLabel,fontsize=12)
        plt.tight_layout()


# block switch plot, target stimuli only
for sessions,sessionsLbl in zip((preHabSessions,habSessions,ephysSessions),('pre-habituation','habituation','ephys')):
    if sessionsLbl!='ephys':
        continue
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)    
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
        y = []
        for sessionTables in sessions: 
            y.append([])
            for d in sessionTables:
                trialBlock = np.array(d['block_index'])
                trialResp = np.array(d['is_response'])
                trialStim = np.array(d['stim_name']) 
                goStim = np.array(d['is_go'])
                nogoStim = np.array(d['is_nogo'])
                targetStim = np.array(d['is_vis_target'] | d['is_aud_target'])
                autoReward = np.array(d['is_reward_scheduled'])
                for blockInd in np.unique(trialBlock): # range(1,6):
                    rewStim = trialStim[(trialBlock==blockInd) & goStim][0]
                    nonRewStim = trialStim[(trialBlock==blockInd) & nogoStim & targetStim][0]
                    if blockInd > 0 and rewStim == blockRewardStim:
                        stim = nonRewStim if 'unrewarded' in stimLbl else rewStim
                        trials = (trialStim==stim) & ~autoReward
                        y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                        pre = trialResp[(trialBlock==blockInd-1) & trials]
                        i = min(preTrials,pre.size)
                        y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                        post = trialResp[(trialBlock==blockInd) & trials]
                        i = min(postTrials,post.size)
                        y[-1][-1][preTrials+1:preTrials+1+i] = post[:i]
            y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(np.arange(-20,21,5))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch (excluding auto-rewards)',fontsize=12)
    ax.set_ylabel('Response rate',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    ax.set_title(sessionsLbl+' ('+str(len(y))+' mice',fontsize=12)
    plt.tight_layout()





