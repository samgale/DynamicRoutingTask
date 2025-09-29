#%%
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import npc_sessions


#%%
control = ['728053_2024-11-18', '728053_2024-11-20','774470_2025-07-29','782106_2025-07-07','782106_2025-07-14','767405_2025-05-28','767405_2025-05-19','739828_2025-02-10','739828_2025-02-17','772657_2025-02-24', '772657_2025-03-03', '774916_2025-03-24', '774916_2025-03-31']
muscimol = ['728053_2024-11-19', '728053_2024-11-21','774470_2025-07-30','782106_2025-07-08', '782106_2025-07-15','767405_2025-05-29','767405_2025-05-20','739828_2025-02-11','739828_2025-02-18', '772657_2025-02-25', '772657_2025-03-04','774916_2025-03-25', '774916_2025-04-01']

mice = list(set([i[:6] for i in muscimol]))


#%%
trialsDf = {lbl: {m: {} for m in mice} for lbl in ('control','muscimol')}
for lbl in trialsDf:
    for m in trialsDf[lbl]:
        d = control if lbl=='control' else muscimol
        sessions = [s for s in d if s[:6]==m]
        for s in sessions:
            trialsDf[lbl][m][s] = npc_sessions.DynamicRoutingSession(s).trials[:]


#%%
stimNames = ('vis1','sound1','vis2','sound2')
stimLabels = ('visual target','auditory target','visual non-target','auditory non-target')
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1) 
for lbl in trialsDf:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,-1],width=5,height=2,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'gmgm',('-','-','--','--')):
            y = []
            for m in trialsDf[lbl]:
                y.append([])
                for s in trialsDf[lbl][m]:
                    df = trialsDf[lbl][m][s]
                    trials = df.stim_name == stim
                    r = df.is_response
                    for blockInd in range(6):
                        blockTrials = df.block_index == blockInd
                        blockRewStim = 'vis1' if df.context_name[blockTrials].iloc[0] == 'vis' else 'sound1'
                        if blockInd > 0 and blockRewStim == rewardStim:
                            y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                            pre = r[(df.block_index==blockInd-1) & trials]
                            i = min(preTrials,pre.size)
                            y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                            post = r[(df.block_index==blockInd) & trials]
                            if stim == blockRewStim:
                                i = min(postTrials,post.size)
                                y[-1][-1][preTrials:preTrials+i] = post[:i]
                            else:
                                i = min(postTrials-5,post.size)
                                y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
            ax.plot(x[preTrials:],m[preTrials:],ls=ls,color=clr)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks([-5,-1,5,9,14,19])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.02])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
        ax.set_ylabel('Response rate',fontsize=12)
        ax.set_title(lbl + ', ' + blockLabel,fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        plt.tight_layout()


#%%
stimNames = ('vis1','sound1')
stimLabels = ('visual target','auditory target')
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1) 
for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded block','auditory rewarded block')):
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    ax.add_patch(matplotlib.patches.Rectangle([-0.5,-1],width=5,height=2,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for lbl,ls,alphaLine,alphaFill in zip(trialsDf,('-','--'),(1,0.5),(0.5,0.25)):
        for stim,stimLbl,clr in zip(stimNames,stimLabels,'gm'):
            y = []
            for m in trialsDf[lbl]:
                y.append([])
                for s in trialsDf[lbl][m]:
                    df = trialsDf[lbl][m][s]
                    trials = df.stim_name == stim
                    r = df.is_response
                    for blockInd in range(6):
                        blockTrials = df.block_index == blockInd
                        blockRewStim = 'vis1' if df.context_name[blockTrials].iloc[0] == 'vis' else 'sound1'
                        if blockInd > 0 and blockRewStim == rewardStim:
                            y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                            pre = r[(df.block_index==blockInd-1) & trials]
                            i = min(preTrials,pre.size)
                            y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                            post = r[(df.block_index==blockInd) & trials]
                            if stim == blockRewStim:
                                i = min(postTrials,post.size)
                                y[-1][-1][preTrials:preTrials+i] = post[:i]
                            else:
                                i = min(postTrials-5,post.size)
                                y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,alpha=alphaLine,label=stimLbl+' ('+lbl+')')
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=alphaFill,lw=0)
            ax.plot(x[preTrials:],m[preTrials:],ls=ls,color=clr,alpha=alphaLine)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=alphaFill,lw=0)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=16)
    ax.set_xticks([-5,-1,5,9,14,19])
    ax.set_xticklabels([-5,-1,1,5,10,15])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials-0.5])
    ax.set_ylim([0,1.02])
    ax.set_xlabel('Trials of indicated type after block switch',fontsize=18)
    ax.set_ylabel('Response rate',fontsize=18)
    ax.set_title('transition to '+blockLabel,fontsize=18)
    ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    plt.tight_layout()

# %%
def calcDprime(hitRate,falseAlarmRate,goTrials,nogoTrials):
    hr = adjustResponseRate(hitRate,goTrials)
    far = adjustResponseRate(falseAlarmRate,nogoTrials)
    z = [scipy.stats.norm.ppf(r) for r in (hr,far)]
    return z[0]-z[1]


def adjustResponseRate(r,n):
    if r == 0:
        r = 0.5/n
    elif r == 1:
        r = 1 - 0.5/n
    return r

#%%
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = (-4,4)
ax.plot(alim,alim,'k:',alpha=0.5)
for k,m in enumerate(mice):
    x = [[],[]]
    y = [[],[]]
    for d,lbl in zip((x,y),trialsDf):
        for s in trialsDf[lbl][m]:
            df = trialsDf[lbl][m][s]
            for i,context in enumerate(('is_vis_context','is_aud_context')):
                trials = df[context] & ~df.is_reward_scheduled
                vis,aud = (trials & df.is_vis_target,trials & df.is_aud_target)
                rew,nonRew = (vis,aud) if context=='is_vis_context' else (aud,vis)
                hn = df.is_response[rew].size
                hr = df.is_response[rew].sum() / hn
                fan = df.is_response[nonRew].size
                far = df.is_response[nonRew].sum() / fan
                d[i].append(calcDprime(hr,far,hn,fan))
    for i,(lbl,clr) in enumerate(zip(('Vis rewarded','Aud rewarded'),('k','tab:orange'))):
        lbl = None if k>0 else lbl
        ax.plot(np.mean(x[i]),np.mean(y[i]),'o',color=clr,ms=10,alpha=0.5,label=lbl)
    ax.plot([np.mean(d) for d in x],[np.mean(d) for d in y],'k-',alpha=0.5)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(np.arange(-4,5,2))
ax.set_yticks(np.arange(-4,5,2))
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('d\' control', fontsize=16)
ax.set_ylabel('d\' muscimol',fontsize=16)
ax.legend(fontsize=14)
plt.tight_layout()


# %%
