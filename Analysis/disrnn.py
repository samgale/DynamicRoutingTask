# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 12:40:00 2026

@author: svc_ccg
"""

import copy
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from disrnnHPC import getData
from disentangled_rnns.library import disrnn


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Sam"


# get session data
trainingPhases = ('initial training','after learning','noAR')
sessionData = {}
testIndex = {}
trainIndex = {}
for phase in trainingPhases:
    sessionData[phase],testIndex[phase],trainIndex[phase] = getData(phase)


# get model data
latentPenalties = {'gru': [None], 'disrnn': [0.01,0.001,0.0001,0.00001,0.000001,0.0000001]}
updatePenalties = {'gru': [None], 'disrnn': [0.01,0.005,0.001,0.0005,0.0001]}
nReps = 3

modelData = {phase: {modelType: {latPenInd: {updPenInd: [None for _ in range(nReps)] for updPenInd in range(len(updatePenalties[modelType]))} for latPenInd in range(len(latentPenalties[modelType]))} for modelType in ('gru','disrnn')} for phase in trainingPhases}
dirPath = os.path.join(baseDir,'DisRNNmodel')
filePaths = glob.glob(os.path.join(dirPath,'*.npz'))
for fileInd,f in enumerate(filePaths):
    print(fileInd)
    fileParts = os.path.splitext(os.path.basename(f))[0].split('_')
    trainingPhase,modelType,latPenInd,updPenInd,rep = fileParts
    latPenInd = int(latPenInd[-1])
    updPenInd = int(updPenInd[-1])
    rep = int(rep[-1])
    with np.load(f,allow_pickle=True) as data:
        d = {key: val for key,val in data.items()}
        d['latentStd'] = d['latentStates'][:,int(0.5*d['latentStates'].shape[1]):,:].std(axis=(0,1))
        d['latentOrder'] = np.argsort(d['latentStd'])[::-1]
        modelData[trainingPhase][modelType][latPenInd][updPenInd][rep] = d


# plot train/test loss trajectory
trainingStage = 'after learning'
stepSize = 10
for losses in ('warmupLosses','modelLosses'):
    fig = plt.figure(figsize=(10,6))
    gs = matplotlib.gridspec.GridSpec(len(latentPenalties['disrnn']),len(updatePenalties['disrnn'])*(nReps+1)-1)
    for i,latPen in enumerate(latentPenalties['disrnn']):
        row = i
        col = -2
        for j,updPen in enumerate(updatePenalties['disrnn']):
            col += 1
            for rep in range(nReps):
                col += 1
                ax = fig.add_subplot(gs[row,col])
                d = modelData[trainingPhase]['disrnn'][i][j][rep]
                for loss,clr in zip(('training_loss','validation_loss'),'kr'):
                    y = d[losses].item()[loss]
                    ax.plot(np.arange(0,y.size*stepSize,stepSize),y,color=clr)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim([0,1])
                if rep==1:
                    ax.set_title('latent penalty: '+str(latPen)+', update penalty: '+str(updPen),fontsize=6)
    plt.tight_layout()


# plot penalized loss, likelihood, and number of open bottlenecks
for trainingPhase in trainingPhases:
    lossMat = np.zeros((len(latentPenalties['disrnn']),len(updatePenalties['disrnn'])))
    likelihoodMat = lossMat.copy()
    nOpenLatentBottlenecks = lossMat.copy()
    nOpenUpdateBottlenecks = lossMat.copy()
    openBottleneckThresh = 0.7
    latentVarThresh = 0.05
    for i,latPen in enumerate(latentPenalties['disrnn']):
        for j,updPen in enumerate(updatePenalties['disrnn']):
            for rep in range(nReps):
                d = modelData[trainingPhase]['disrnn'][i][j][rep]
                if len(d) > 0:
                    lossMat[i,j] += d['modelLosses'].item()['validation_loss'][-1]
                    likelihoodMat[i,j] += np.mean(d['likelihood'])
                    params = d['modelParams'].item()['hk_disentangled_rnn']
                    nLat,nUpdObs,nUpdLat = [np.sum((abs(params[key])<openBottleneckThresh) & (d['latentVar']>latentVarThresh)) for key in ('latent_sigma_params','update_net_obs_sigma_params','update_net_latent_sigma_params')]
                    nOpenLatentBottlenecks[i,j] += nLat
                    nOpenUpdateBottlenecks[i,j] += (nUpdObs + nUpdLat) / nLat
    lossMat /= nReps
    likelihoodMat /= nReps
    nOpenLatentBottlenecks /= nReps
    nOpenUpdateBottlenecks /= nReps
    
    for m,lbl in zip((lossMat,likelihoodMat,nOpenLatentBottlenecks,nOpenUpdateBottlenecks),('penalized loss','likelihood','# open latent bottlenecks','# open update bottlenecks per open latent')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(m,cmap='magma')
        cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
        # cb.set_ticks((0,0.2,0.4,0.6))
        # cb.set_ticklabels((0,0.2,0.4,0.6),fontsize=12)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',labelsize=12)
        ax.set_xticks(np.arange(len(updatePenalties['disrnn'])))
        ax.set_yticks(np.arange(len(latentPenalties['disrnn'])))
        ax.set_xticklabels(updatePenalties['disrnn'])
        ax.set_yticklabels(latentPenalties['disrnn'])
        ax.set_xlabel('Update penalty',fontsize=14)
        ax.set_ylabel('Latent penalty',fontsize=14)
        ax.set_title(lbl,fontsize=14)
        plt.tight_layout()
        

# compare disrnn and gru



# plot bottleneck structure
for trainingPhase in trainingPhases:
    fig = plt.figure(figsize=(16,10))
    gs = matplotlib.gridspec.GridSpec(len(latentPenalties['disrnn']),len(updatePenalties['disrnn'])*(nReps+1)-1)
    for i,latPen in enumerate(latentPenalties['disrnn']):
        row = i
        col = -2
        for j,updPen in enumerate(updatePenalties['disrnn']):
            col += 1
            for rep in range(nReps):
                col += 1
                d = modelData[trainingPhase]['disrnn'][i][j][rep]
                if len(d) > 0:
                    params = d['modelParams'].item()['hk_disentangled_rnn']
                    config = d['modelConfig'].item()
                    latentOrder = d['latentOrder']
                    update_input_names = config.x_names
                    latent_names = ['latent '+str(ln) for ln in np.arange(1,config.latent_size + 1)]
                    update_obs_sigmas_t = np.transpose(disrnn.reparameterize_sigma(params['update_net_obs_sigma_params']))
                    update_latent_sigmas_t = np.transpose(disrnn.reparameterize_sigma(params['update_net_latent_sigma_params']))
                    update_sigmas = np.concatenate((update_obs_sigmas_t, update_latent_sigmas_t), axis=1)
                    choice_sigmas = np.array(disrnn.reparameterize_sigma(np.transpose(params['choice_net_sigma_params'])))
                    update_sigma_order = np.concatenate((np.arange(0,len(update_input_names),1),len(update_input_names) + latentOrder),axis=0)
                    update_sigmas = update_sigmas[latentOrder,:]
                    update_sigmas = update_sigmas[:,update_sigma_order]
                    
                    ax = fig.add_subplot(gs[row,col])
                    im = ax.imshow(1 - update_sigmas,clim=(0,1),cmap='Oranges')
                    for side in ('right','top','left','bottom'):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(direction='out')
                    ax.set_xticks([])
                    ax.set_yticks([])
    #                 ax.set_xticks(np.arange(len(update_input_names) + len(latent_names)))
    #                 ax.set_yticks(np.arange(len(latent_names)))
    #                 ax.set_xticklabels([])
    #                 ax.set_yticklabels([])
    #                 if row==0 and col==0:
    #                     ax.set_yticklabels(latent_names)
    #                 if i==len(latentPenalties['disrnn'])-1 and col==1:
    #                     ax.set_xticklabels(update_input_names + latent_names,rotation='vertical')
                    if rep==1:
                        ax.set_title('latent penalty: '+str(latPen)+', update penalty: '+str(updPen),fontsize=6)
    plt.tight_layout()    


# choose network to plot
trainingPhase = 'initial training'
latPenInd = 4
updPenInd = 2
rep = 0
nLatents = 5
d = modelData[trainingPhase]['disrnn'][latPenInd][updPenInd][rep]

trainingPhase = 'after learning'
latPenInd = 0
updPenInd = 2
rep = 0
nLatents = 2
d = modelData[trainingPhase]['disrnn'][latPenInd][updPenInd][rep]


# plot latent states for one session
sessionInd = 100
for lat,latInd in enumerate(d['latentOrder'][:nLatents]):
    fig = plt.figure(figsize=(10,5))
    obj = np.array(sessionData[trainingPhase])[testIndex[trainingPhase]][sessionInd]
    state = d['latentStates'][sessionInd][:obj.nTrials,latInd]   
    ylim = (-1.9,1.9)
    ax = fig.add_subplot(1,1,1)
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
        blockTrials = obj.trialBlock==blockInd+1
        blockStart,blockEnd = np.where(blockTrials)[0][[0,-1]]
        if rewStim=='vis1':
            ax.add_patch(matplotlib.patches.Rectangle([blockStart+0.5,ylim[0]],width=blockEnd-blockStart+1,height=ylim[1]-ylim[0],facecolor='0.75',edgecolor=None,zorder=0))
    ax.plot(np.arange(obj.nTrials)+1,state,'k')
    for stim,clr in zip(('vis1','sound1'),'gm'):
        trials = (obj.trialStim==stim) & obj.trialResponse
        ax.plot(np.where(trials)[0]+1,state[trials],'o',mec=clr,mfc='none')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,obj.nTrials+1])
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial',fontsize=14)
    ax.set_ylabel('Latent '+str(lat+1)+' state',fontsize=14)
    plt.tight_layout()


# plot update rules
stimNames = ('vis1','vis2','sound1','sound2')
prevState = []
updatedState = []
for lat,latInd in enumerate(d['latentOrder'][:nLatents]):
    prevState.append({rewStim: {stim: {resp: {ar: [] for ar in (0,1)} for resp in (0,1)} for stim in stimNames} for rewStim in ('vis1','sound1')})
    updatedState.append(copy.deepcopy(prevState[-1]))
    for obj,state in zip(np.array(sessionData[trainingPhase])[testIndex[trainingPhase]],d['latentStates']):
        state = state[:obj.nTrials,latInd]
        for rewStim in prevState[-1]:
            for stim in stimNames:
                for resp in (0,1):
                    for ar in (0,1):
                        trials = np.where((obj.rewardedStim==rewStim) & (obj.trialStim==stim) & (obj.trialResponse==resp) & (obj.autoRewarded==ar))[0]
                        trials = trials[trials < obj.nTrials-1]
                        for tr in trials:
                            prevState[-1][rewStim][stim][resp][ar].append(state[tr])
                            updatedState[-1][rewStim][stim][resp][ar].append(state[tr+1])

for lat,(ps,us) in enumerate(zip(prevState,updatedState)):
    fig = plt.figure(figsize=(8,4.5))
    gs = matplotlib.gridspec.GridSpec(2,4)
    fig.text(0.5,1,'Vis rewarded',ha='center',va='top',fontsize=12)
    fig.text(0.5,0.5,'Aud rewarded',ha='center',va='top',fontsize=12)
    fig.text(0.5,0,'Previous Latent '+str(lat+1),ha='center',fontsize=12)
    fig.text(0,0.5,'Updated Latent '+str(lat+1),rotation='vertical',va='center',fontsize=12)
    alim = [-1.9,1.9]
    for row,rewStim in enumerate(ps):
        for col,stim in enumerate(ps[rewStim]):
                ax = fig.add_subplot(gs[row,col])
                ax.plot(alim,alim,'--',color='0.5')
                for resp in ps[rewStim][stim]:
                    for ar in ps[rewStim][stim][resp]:
                        if resp and ar:
                            continue
                        elif resp:
                            clr = 'g'
                            lbl = 'response'
                        elif ar:
                            clr = 'b'
                            lbl = 'non-contingent reward'
                        else:
                            clr = 'r'
                            lbl = 'no response'
                        ax.plot(ps[rewStim][stim][resp][ar],us[rewStim][stim][resp][ar],'o',mec=clr,mfc='none',ms=2,alpha=0.2,label=(lbl if row==1 and col==0 else None))
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False,labelsize=8)
                if row < 1:
                    ax.set_xticklabels([])
                if col > 0:
                    ax.set_yticklabels([])
                if row == 0:
                    ax.set_title(stim,fontsize=10)
                if row == 1 and col == 0:
                    ax.legend(fontsize=6)
                ax.set_xlim(alim)
                ax.set_ylim(alim)
                ax.set_aspect('equal')
    plt.tight_layout()
    

# plot choice rule for single latent
lat = 1
latInd = d['latentOrder'][lat]
x = [[] for _ in stimNames]
y = copy.deepcopy(x)
for obj,state,pr in zip(np.array(sessionData[trainingPhase])[testIndex[trainingPhase]],d['latentStates'],d['probResp']):
    for i,stim in enumerate(stimNames):
        trials = obj.trialStim==stim
        x[i].append(state[:obj.nTrials,latInd][trials])
        y[i].append(pr[:obj.nTrials][trials])
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
stimNames = ('vis1','vis2','sound1','sound2')
stimColors = 'rmbc'
binSize = 0.1
bins = np.arange(-2,2+binSize,binSize)
for i,(stim,clr,lbl) in enumerate(zip(stimNames,stimColors,('VIS+','VIS-','AUD+','AUD-'))):
    xi = np.concatenate(x[i])
    yi = np.concatenate(y[i])
    ax.plot(xi,yi,'o',mec=clr,mfc='none',alpha=0.1,label=lbl)
    ind = np.digitize(xi,bins)
    bx = np.unique(ind)
    m = [np.nanmean(yi[ind==b]) for b in bx]
    ax.plot(bins[bx],m,color=clr,lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlabel('Latent '+str(lat+1)+' state',fontsize=14)
ax.set_ylabel('Prob resp',fontsize=14)
ax.legend()
plt.tight_layout()


# plot choice rule for two latents
stimNames = ('vis1','vis2','sound1','sound2')
lat = [1,3]
latInd = d['latentOrder'][lat]
binSize = 0.1
bins = np.arange(-2,2+binSize,binSize)
z = [[] for _ in stimNames]
n = copy.deepcopy(z)
for obj,state,pr in zip(np.array(sessionData[trainingPhase])[testIndex[trainingPhase]],d['latentStates'],d['probResp']):
    for i,stim in enumerate(stimNames):
        trials = obj.trialStim==stim
        x,y = (state[:obj.nTrials,latInd][trials]).T
        z[i].append(np.histogram2d(x,y,bins=bins,weights=pr[:obj.nTrials][trials])[0])
        n[i].append(np.histogram2d(x,y,bins)[0])

zavg = [np.sum(w,axis=0) / np.sum(c,axis=0) for w,c in zip(z,n)]

for i,(stim,lbl) in enumerate(zip(stimNames,('VIS+','VIS-','AUD+','AUD-'))):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(zavg[i],cmap='magma',clim=(0,1),extent=(bins[0],bins[-1],bins[0],bins[-1]),origin='lower')
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',labelsize=10)
    ax.set_xlabel('Latent '+str(lat[1]+1)+' state',fontsize=12)
    ax.set_ylabel('Latent '+str(lat[0]+1)+' state',fontsize=12)
    ax.set_title('Prob. response ('+lbl+')',fontsize=14)
    plt.tight_layout()

            
# block transition plot
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for src in ('mice','model prediction','model simulation'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
        y = []
        for sessionInd,obj in enumerate(np.array(sessionData[trainingPhase])[testIndex[trainingPhase]]):
            y.append([])
            if src == 'mice':
                resp = obj.trialResponse
            elif src == 'model prediction':
                resp = d['probResp'][sessionInd][:obj.nTrials]
            elif src == 'model simulation':
                resp = np.mean([s for s in d['simProbResp'][sessionInd]],axis=0)
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                if blockInd > 0:
                    stim = np.setdiff1d(obj.blockStimRewarded,rewStim)[0] if 'unrewarded' in stimLbl else rewStim
                    if 'non-target' in stimLbl:
                        stim = stim[:-1]+'2'
                    trials = (obj.trialStim==stim)
                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                    pre = resp[(obj.trialBlock==blockInd) & trials]
                    i = min(preTrials,pre.size)
                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                    post = resp[(obj.trialBlock==blockInd+1) & trials]
                    if stim==rewStim:
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
        ax.plot(x[preTrials:],m[preTrials:],color=clr,ls=ls)
        ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks([-5,-1,5,9,14,19])
    ax.set_xticklabels([-5,-1,1,5,10,15])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials-0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials after block switch',fontsize=16)
    ax.set_ylabel('Response rate',fontsize=16)
    ax.set_title(src)
    #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
    plt.tight_layout()
    
# by block type
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for src in ('mice','model prediction','model simulation'):
    for blockRew in ('vis1','sound1'):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
        for stim,clr,ls in zip(('vis1','vis2','sound1','sound2'),'ggmm',('-','--','-','--')):
            y = []
            for sessionInd,obj in enumerate(np.array(sessionData)[testIndex]):
                y.append([])
                if src == 'mice':
                    resp = obj.trialResponse
                elif src == 'model prediction':
                    resp = probResp[modelType][latPenInd][updPenInd][sessionInd][:obj.nTrials]
                elif src == 'model simulation':
                    resp = simResp[modelType][latPenInd][updPenInd][sessionInd][:obj.nTrials]
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if rewStim==blockRew and blockInd > 0:
                        trials = (obj.trialStim==stim)
                        y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                        pre = resp[(obj.trialBlock==blockInd) & trials]
                        i = min(preTrials,pre.size)
                        y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                        post = resp[(obj.trialBlock==blockInd+1) & trials]
                        if stim==rewStim:
                            i = min(postTrials,post.size)
                            y[-1][-1][preTrials:preTrials+i] = post[:i]
                        else:
                            i = min(postTrials-5,post.size)
                            y[-1][-1][preTrials+5:preTrials+5+i] = post[:i]
                y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x[:preTrials],m[:preTrials],color=clr,ls=ls,label=stim)
            ax.fill_between(x[:preTrials],(m+s)[:preTrials],(m-s)[:preTrials],color=clr,alpha=0.25)
            ax.plot(x[preTrials:],m[preTrials:],color=clr,ls=ls)
            ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        ax.set_xticks([-5,-1,5,9,14,19])
        ax.set_xticklabels([-5,-1,1,5,10,15])
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials-0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials after block switch',fontsize=16)
        ax.set_ylabel('Response rate',fontsize=16)
        ax.set_title(src)
        #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
        plt.tight_layout()

# first block
preTrials = 0
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for src in ('mice','model prediction','model simulation'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
        y = []
        for sessionInd,obj in enumerate(np.array(sessionData)[testIndex]):
            y.append([])
            if src == 'mice':
                resp = obj.trialResponse
            elif src == 'model prediction':
                resp = probResp[modelType][latPenInd][updPenInd][sessionInd][:obj.nTrials]
            elif src == 'model simulation':
                resp = simResp[modelType][latPenInd][updPenInd][sessionInd][:obj.nTrials]
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                if blockInd == 0:
                    stim = np.setdiff1d(obj.blockStimRewarded,rewStim)[0] if 'unrewarded' in stimLbl else rewStim
                    if 'non-target' in stimLbl:
                        stim = stim[:-1]+'2'
                    trials = (obj.trialStim==stim)
                    y[-1].append(np.full(preTrials+postTrials+1,np.nan))
                    pre = resp[(obj.trialBlock==blockInd) & trials]
                    i = min(preTrials,pre.size)
                    y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                    post = resp[(obj.trialBlock==blockInd+1) & trials]
                    if stim==rewStim:
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
        ax.plot(x[preTrials:],m[preTrials:],color=clr,ls=ls)
        ax.fill_between(x[preTrials:],(m+s)[preTrials:],(m-s)[preTrials:],color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xticks([-5,-1,5,9,14,19])
    ax.set_xticklabels([-5,-1,1,5,10,15])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials-0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials after block switch',fontsize=16)
    ax.set_ylabel('Response rate',fontsize=16)
    ax.set_title(src)
    #ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=18)
    plt.tight_layout()       


# intra-block resp correlations
def getBlockTrials(obj,block,epoch):
    blockTrials = (obj.trialBlock==block) & ~obj.autoRewardScheduled
    n = blockTrials.sum()
    half = int(n/2)
    startTrial = half if epoch=='last half' else 0
    endTrial = half if epoch=='first half' else n
    return np.where(blockTrials)[0][startTrial:endTrial]


def detrend(r,order=2):
    x = np.arange(r.size)
    return r - np.polyval(np.polyfit(x,r,order),x)


def getCorrelation(r1,r2,rs1,rs2,corrSize=200,detrendOrder=None):
    if detrendOrder is not None:
        r1 = detrend(r1,detrendOrder)
        r2 = detrend(r2,detrendOrder)
        rs1 = rs1.copy()
        rs2 = rs2.copy()
        for z in range(rs1.shape[1]):
            rs1[:,z] = detrend(rs1[:,z],detrendOrder)
            rs2[:,z] = detrend(rs2[:,z],detrendOrder)
    c = np.correlate(r1,r2,'full') / (np.linalg.norm(r1) * np.linalg.norm(r2))   
    cs = np.mean([np.correlate(rs1[:,z],rs2[:,z],'full') / (np.linalg.norm(rs1[:,z]) * np.linalg.norm(rs2[:,z])) for z in range(rs1.shape[1])],axis=0)
    n = c.size // 2 + 1
    corrRaw = np.full(corrSize,np.nan)
    corrRaw[:n] = c[-n:]
    corr = np.full(corrSize,np.nan)
    corr[:n] = (c-cs)[-n:] 
    return corr,corrRaw

epoch = 'full'
stimNames = ('vis1','sound1','vis2','sound2')
autoCorrMat = {src: np.zeros((4,1,100)) for src in ('mice','model')}
autoCorrDetrendMat = copy.deepcopy(autoCorrMat)
corrWithinMat = {src: np.zeros((4,4,1,200)) for src in ('mice','model')}
corrWithinDetrendMat = copy.deepcopy(corrWithinMat)
minTrials = 3
nShuffles = 10
for src in ('mice','model'):
    autoCorr = [[] for _ in range(4)]
    autoCorrDetrend = copy.deepcopy(autoCorr)
    corrWithin = [[[] for _ in range(4)] for _ in range(4)]
    corrWithinDetrend = copy.deepcopy(corrWithin)
    for sessionInd,obj in enumerate(np.array(sessionData)[testIndex]):
        if src=='mice': 
            trialResponse = [obj.trialResponse]
        else:    
            trialResponse = [s for s in d['simResp'][sessionInd]]
        for tr in trialResponse:
            resp = np.zeros((4,obj.nTrials))
            respShuffled = np.zeros((4,obj.nTrials,nShuffles))
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                    stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                    if len(stimTrials) < minTrials:
                        continue
                    r = tr[stimTrials].astype(float)
                    r[r<1] = -1
                    resp[i,stimTrials] = r
                    for z in range(nShuffles):
                        respShuffled[i,stimTrials,z] = np.random.permutation(r)
            
            for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                blockTrials = getBlockTrials(obj,blockInd+1,epoch)
                for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2')):
                    stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                    if len(stimTrials) < minTrials:
                        continue
                    r = resp[i,stimTrials]
                    rs = respShuffled[i,stimTrials]
                    corr,corrRaw = getCorrelation(r,r,rs,rs,100)
                    autoCorr[i].append(corr)
                    corrDetrend,corrRawDetrend = getCorrelation(r,r,rs,rs,100,detrendOrder=2)
                    autoCorrDetrend[i].append(corrDetrend)
                
                r = resp[:,blockTrials]
                rs = respShuffled[:,blockTrials]
                for i,(r1,rs1) in enumerate(zip(r,rs)):
                    for j,(r2,rs2) in enumerate(zip(r,rs)):
                        if np.count_nonzero(r1) >= minTrials and np.count_nonzero(r2) >= minTrials:
                            corr,corrRaw = getCorrelation(r1,r2,rs1,rs2)
                            corrWithin[i][j].append(corr)
                            corrDetrend,corrRawDetrend = getCorrelation(r1,r2,rs1,rs2,detrendOrder=2)
                            corrWithinDetrend[i][j].append(corrDetrend)
    
    m = 0
    autoCorrMat[src][:,m] = np.nanmean(autoCorr,axis=1)
    autoCorrDetrendMat[src][:,m] = np.nanmean(autoCorrDetrend,axis=1)
            
    corrWithinMat[src][:,:,m] = np.nanmean(corrWithin,axis=2)
    corrWithinDetrendMat[src][:,:,m] = np.nanmean(corrWithinDetrend,axis=2)

stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)')


fig = plt.figure(figsize=(12,10))       
gs = matplotlib.gridspec.GridSpec(4,4)
x = np.arange(1,200)
for i,ylbl in enumerate(stimLabels):
    for j,xlbl in enumerate(stimLabels[:4]):
        ax = fig.add_subplot(gs[i,j])
        for lbl,clr in zip(('mice','model'),'kr'):
            mat = corrWithinDetrendMat[lbl][i,j,:,1:]
            m = np.nanmean(mat,axis=0)
            s = np.nanstd(mat,axis=0) / (len(mat) ** 0.5)
            ax.plot(x,m,clr,label=lbl)
            ax.fill_between(x,m-s,m+s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,20])
        ax.set_ylim([-0.03,0.05])
        if i==3:
            ax.set_xlabel('Lag (trials)',fontsize=14)
        else:
            ax.set_xticklabels([])
        if j==0:
            ax.set_ylabel(ylbl,fontsize=14)
        else:
            ax.set_yticklabels([])
        if i==0:
            ax.set_title(xlbl,fontsize=14)
        if i==0 and j==3:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
plt.tight_layout()





