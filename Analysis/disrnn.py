# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 12:40:00 2026

@author: svc_ccg
"""

import copy
import os
import typing
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getSessionsToPass,getRNNSessions,getSessionData

from disentangled_rnns.library import disrnn
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import two_armed_bandits
from disentangled_rnns.library import plotting
import haiku as hk
import jax
import optax


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"


class DynamicRoutingEnvironment(two_armed_bandits.BaseEnvironment):

    def __init__(self,
                 networkInput: np.ndarray, # DatasetRNN._xs for one session (ntrials x 1 x ninputs) 
                 trialStim: np.ndarray,
                 rewardedStim: np.ndarray,
                 rewardScheduled: np.ndarray,
                 seed: typing.Optional[int] = None,
                 n_arms: int = 2):

        super().__init__(seed=seed, n_arms=n_arms)
        
        self.xs = networkInput.copy()
        self.trialStim = trialStim
        self.rewardedStim = rewardedStim
        self.rewardScheduled = rewardScheduled
        self.new_session()
      
    def new_session(self):
        pass
      
    def step(self, attempted_choice: int, trial_index: int):
        choice = attempted_choice
        instructed = self.rewardScheduled[trial_index]
        reward = (choice and self.trialStim[trial_index] == self.rewardedStim[trial_index]) or instructed
        if trial_index < self.trialStim.size - 1:
            xs = self.xs[trial_index + 1]
            xs[0,4] = choice
            xs[0,5] = reward
        else:
            xs = None
        return choice, reward, xs



# get data for pooled training
summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
isStandardRegimen = getIsStandardRegimen(summaryDf)
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])

sessionDataByMouse = {phase: [] for phase in ('initial training','after learning')}
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
    sessions = np.where(sessions)[0]
    sessionsToPass = getSessionsToPass(mouseId,df,sessions,stage=5)
    sessionDataByMouse['initial training'].append([getSessionData(mouseId,startTime,lightLoad=True) for startTime in df.loc[sessions[:2],'start time']])
    sessionDataByMouse['after learning'].append([getSessionData(mouseId,startTime,lightLoad=True) for startTime in df.loc[sessions[sessionsToPass:sessionsToPass+2],'start time']])
    
trainingPhase = 'after learning'
sessionData = (# first session from odd mice, second session from even mice
               [d[0] for d in sessionDataByMouse[trainingPhase][::2]] + [d[1] for d in sessionDataByMouse[trainingPhase][1::2]]
               # second session from odd mice, first session from even mice
               + [d[1] for d in sessionDataByMouse[trainingPhase][::2]] + [d[0] for d in sessionDataByMouse[trainingPhase][1::2]])
testIndex = np.arange(len(mice))
trainIndex = np.arange(len(mice),2*len(mice))
    
    

def getDisrnnDataset(sessionData,testIndex):
    nInputs = 6
    stimNames = ('vis1','vis2','sound1','sound2')
    maxTrials = max(session.nTrials for session in sessionData)
    modelInput = -1 * np.ones((maxTrials,len(sessionData),nInputs),dtype=np.float32)
    targetOutput = -1 * np.ones((maxTrials,len(sessionData),1),dtype=np.int32)
    for i,session in enumerate(sessionData):
        n = session.nTrials
        for j,stim in enumerate(stimNames):    
            modelInput[:n,i,j] = session.trialStim == stim
        modelInput[0,i,4:6] = 0
        modelInput[1:n,i,4] = session.trialResponse[:-1]
        modelInput[1:n,i,5] = session.trialRewarded[:-1]
        targetOutput[:n,i,0] = session.trialResponse
    
    trainIndex = np.setdiff1d(np.arange(len(sessionData)),testIndex)    
    testDataset,trainDataset = [rnn_utils.DatasetRNN(
            xs=modelInput[:,i],
            ys=targetOutput[:,i],
            y_type='categorical',
            n_classes=2,
            x_names=['vis target','vis non-target','aud target','aud non-target','prev resp','prev outcome'],
            y_names=['resp'],
            batch_size=128,
            batch_mode='random') # random or rolling
        for i in (testIndex,trainIndex)]
    return testDataset,trainDataset


testDataset,trainDataset = getDisrnnDataset(sessionData,testIndex)
modelTypes = ('gru',) # 'gru', 'disrnn'
latentPenalties= {}
updatePenalties = {}
modelParams = {}
modelConfig = {}
latentSigmas = {}
latentOrder = {}
latentStates = {}
probResp = {}
likelihood = {}
for modelType in modelTypes:
    if modelType == 'gru':
        latentPenalties[modelType] = [None]
        updatePenalties[modelType] = [None]
    else:
        latentPenalties[modelType] = [0.03,0.01,0.003]
        updatePenalties[modelType] = [0.03,0.01,0.003]
    modelParams[modelType] = [[] for _ in range(len(latentPenalties[modelType]))]
    modelConfig[modelType] = copy.deepcopy(modelParams[modelType])
    latentSigmas[modelType] = copy.deepcopy(modelParams[modelType])
    latentOrder[modelType] = copy.deepcopy(modelParams[modelType])
    latentStates[modelType] = [[[] for _ in range(len(updatePenalties[modelType]))] for _ in range(len(latentPenalties[modelType]))]
    probResp[modelType] = copy.deepcopy(latentStates[modelType])
    likelihood[modelType] = copy.deepcopy(latentStates[modelType])
    for i,latPen in enumerate(latentPenalties[modelType]):
        for j,updPen in enumerate(updatePenalties[modelType]):
            if modelType == 'disrnn':
                # define the disRNN architecture
                disrnn_config = disrnn.DisRnnConfig(
                    # Dataset related
                    obs_size=6,
                    output_size=2,
                    x_names=testDataset.x_names,
                    y_names=testDataset.y_names,
                    # Network architecture
                    latent_size=6,
                    update_net_n_units_per_layer=16,
                    update_net_n_layers=4,
                    choice_net_n_units_per_layer=4,
                    choice_net_n_layers=2,
                    activation="leaky_relu",
                    # Penalties
                    noiseless_mode=False,
                    latent_penalty=latPen,
                    update_net_obs_penalty=updPen,
                    update_net_latent_penalty=updPen,
                    choice_net_latent_penalty=0.001,
                    l2_scale=1e-3)
                
                # Define a config for warmup training with no noise and no penalties
                disrnn_config_warmup = copy.deepcopy(disrnn_config)
                disrnn_config_warmup.latent_penalty = 0
                disrnn_config_warmup.choice_net_latent_penalty = 0
                disrnn_config_warmup.update_net_obs_penalty = 0
                disrnn_config_warmup.update_net_latent_penalty = 0
                disrnn_config_warmup.l2_scale = 0
                disrnn_config_warmup.noiseless_mode = True
            
                # Define network builder functions
                make_disrnn = lambda: disrnn.HkDisentangledRNN(disrnn_config)
                make_disrnn_warmup = lambda: disrnn.HkDisentangledRNN(disrnn_config_warmup)
                make_network = make_disrnn
                make_eval_network = make_disrnn_warmup
                loss = "penalized_categorical"
            else:
                make_gru = lambda: hk.DeepRNN([hk.GRU(8), hk.Linear(2)])
                make_network = make_gru
                make_eval_network = make_gru
                loss = "categorical"
            
            # Define an optimizer
            opt = optax.adamw(learning_rate=0.001,weight_decay=0.01)
            
            if modelType == 'disrnn':
                # Warmup training with no noise and no penalties
                params, _, _ = rnn_utils.train_network(
                    make_disrnn_warmup,
                    training_dataset=trainDataset,
                    validation_dataset=testDataset,
                    loss=loss,
                    params=None,
                    opt_state=None,
                    opt=opt,
                    n_steps=1000,
                    do_plot=False)
            else:
                params = None
            
            # Additional training using information penalty
            params, _, _ = rnn_utils.train_network(
                make_network,
                training_dataset=trainDataset,
                validation_dataset=testDataset,
                loss=loss,
                params=params,
                opt_state=None,
                opt=opt,
                n_steps=5000,
                do_plot=True)
            
            # store model params
            modelParams[modelType][i].append(params)
            if modelType == 'disrnn':
                modelConfig[modelType][i].append(disrnn_config)
                latentSigmas[modelType][i].append(np.array(disrnn.reparameterize_sigma(params['hk_disentangled_rnn']['latent_sigma_params'])))
                latentOrder[modelType][i].append(np.argsort(latentSigmas[modelType][i][-1]))
            
            # Eval network on unseen data
            # Use the wamrup disrnn so that there will be no noise
            for s in np.arange(len(testIndex)):
                xs = testDataset._xs[:,[s]]
                ys = testDataset._ys[:,[s]]
                network_outputs,network_states = rnn_utils.eval_network(make_eval_network,params,xs)
                latentStates[modelType][i][j].append(network_states[:,0])
                probResp[modelType][i][j].append(np.exp(network_outputs[:,0,1]) / (np.exp(network_outputs[:,0,0]) + np.exp(network_outputs[:,0,1])))
                likelihood[modelType][i][j].append(rnn_utils.normalized_likelihood(ys,network_outputs[:,:,:2]))


# simulate behavior with trained networks
simResp = {}
for modelType in modelTypes:
    simResp[modelType] = [[[] for _ in range(len(updatePenalties[modelType]))] for _ in range(len(latentPenalties[modelType]))]
    for i,latPen in enumerate(latentPenalties[modelType]):
        for j,updPen in enumerate(updatePenalties[modelType]):
            if modelType == 'disrnn':
                disrnn_config= copy.deepcopy(modelConfig[modelType][i][j])
                disrnn_config.latent_penalty = 0
                disrnn_config.choice_net_latent_penalty = 0
                disrnn_config.update_net_obs_penalty = 0
                disrnn_config.update_net_latent_penalty = 0
                disrnn_config.l2_scale = 0
                disrnn_config.noiseless_mode = True
                make_network = lambda: disrnn.HkDisentangledRNN(disrnn_config)
            else:
                make_network = lambda: hk.DeepRNN([hk.GRU(8), hk.Linear(2)])
            agent = two_armed_bandits.AgentNetwork(make_network,modelParams[modelType][i][j])
            for sessionInd,session in enumerate(np.array(sessionData)[testIndex]):
                networkInput = testDataset._xs[:,[sessionInd]]
                env = DynamicRoutingEnvironment(networkInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled)
                d = two_armed_bandits.create_dataset(agent,env,session.nTrials,1)
                simResp[modelType][i][j].append(d._ys[:,0,0])


# plot likelihood
likelihoodMat = np.zeros((len(latentPenalties['disrnn']),len(updatePenalties['disrnn'])))
for i,latPen in enumerate(latentPenalties['disrnn']):
    for j,updPen in enumerate(updatePenalties['disrnn']):
        likelihoodMat[i,j] = np.mean(likelihood['disrnn'][i][j])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(likelihoodMat,cmap='magma')
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
ax.set_title('Likelihood',fontsize=14)
plt.tight_layout()


# plot bottleneck structure
for i,latPen in enumerate(latentPenalties['disrnn']):
    for j,updPen in enumerate(updatePenalties['disrnn']):
        plotting.plot_bottlenecks(modelParams['disrnn'][i][j],modelConfig['disrnn'][i][j])
        

# choose network to plot
latPenInd = 0
updPenInd = 1
nLatents = 4


# plot latent states
sessionInd = 0
for lat,latInd in enumerate(latentOrder['disrnn'][latPenInd][updPenInd][:nLatents]):
    fig = plt.figure(figsize=(10,5))
    obj = np.array(sessionData)[testIndex][sessionInd]
    state = latentStates['disrnn'][latPenInd][updPenInd][sessionInd][:obj.nTrials,latInd]   
    ylim = 1.05 * np.array([state.min(),state.max()])
    ax = fig.add_subplot(1,1,1)
    stimTime = obj.stimStartTimes
    tintp = np.arange(obj.trialEndTimes[-1])
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
        blockTrials = obj.trialBlock==blockInd+1
        blockStart,blockEnd = np.where(blockTrials)[0][[0,-1]]
        if rewStim=='vis1':
            ax.add_patch(matplotlib.patches.Rectangle([blockStart+0.5,ylim[0]],width=blockEnd-blockStart+1,height=ylim[1]-ylim[0],facecolor='0.75',edgecolor=None,zorder=0))
    ax.plot(np.arange(obj.nTrials)+1,state,'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,obj.nTrials+1])
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial',fontsize=14)
    ax.set_ylabel('Latent '+str(lat+1)+' state',fontsize=14)
    plt.tight_layout()


# plot update rules
stimNames = ('vis1','vis2','sound1','sound2','catch')
deltaState = []
cmax = []
for latInd in latentOrder['disrnn'][latPenInd][updPenInd][:nLatents]:
    deltaState.append({rewStim: {resp: [] for resp in (1,0)} for rewStim in ('vis1','sound1')})
    cmax.append(0)
    for rewStim in deltaState[-1]:
        for resp in deltaState[-1][rewStim]:
            for obj,state in zip(np.array(sessionData)[testIndex],latentStates['disrnn'][latPenInd][updPenInd]):
                state = state[:obj.nTrials,latInd]
                ds = np.zeros((len(stimNames),)*2)
                n = ds.copy()
                blockTypeTrials = obj.rewardedStim==rewStim
                for i,stim in enumerate(stimNames):
                    trials = np.where(blockTypeTrials & (obj.trialStim==stim) & ~obj.autoRewardScheduled)[0]
                    trials = trials[trials > 0]
                    for tr in trials:
                        if obj.trialResponse[tr-1]==resp:
                            j = stimNames.index(obj.trialStim[tr-1])
                            ds[i,j] += state[tr] - state[tr-1]
                            n[i,j] += 1
                deltaState[-1][rewStim][resp].append(ds / n)
            deltaState[-1][rewStim][resp] = np.nanmean(deltaState[-1][rewStim][resp],axis=0)
            cmax[-1] = max(cmax[-1],np.max(np.absolute(deltaState[-1][rewStim][resp])))

stimLabels = ('VIS+','VIS-','AUD+','AUD-','catch')
for lat,ds in enumerate(deltaState):
    fig = plt.figure(figsize=(8,6))
    fig.suptitle('change in latent '+str(lat+1),fontsize=14)
    fig.text(0.02,0.2,'aud rewarded',rotation='vertical',fontsize=12)
    fig.text(0.02,0.6,'vis rewarded',rotation='vertical',fontsize=12)
    gs = matplotlib.gridspec.GridSpec(2,2)
    for row,rewStim in enumerate(ds):
        for col,resp in enumerate(ds[rewStim]):
            ax = fig.add_subplot(gs[row,col])
            im = ax.imshow(ds[rewStim][resp],cmap='bwr',clim=(-cmax[lat],cmax[lat]))
            cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',labelsize=8)
            ax.set_xticks(np.arange(len(stimLabels)))
            ax.set_yticks(np.arange(len(stimLabels)))
            if row == 1:
                ax.set_xticklabels(stimLabels,ha='center')
                ax.set_xlabel('previous stim',fontsize=10)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_yticklabels(stimLabels)
                ax.set_ylabel('current stim',fontsize=10)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(('previous trial response' if resp else 'no response'),fontsize=12)
    plt.tight_layout()
    

# plot choice rule for single latent
lat = 0
latInd = latentOrder['disrnn'][latPenInd][updPenInd][lat]
x = [[] for _ in stimNames]
y = copy.deepcopy(x)
for obj,state,pr in zip(np.array(sessionData)[testIndex],latentStates['disrnn'][latPenInd][updPenInd],probResp['disrnn'][latPenInd][updPenInd]):
    for i,stim in enumerate(stimNames):
        trials = obj.trialStim==stim
        x[i].append(state[:obj.nTrials,latInd][trials])
        y[i].append(pr[:obj.nTrials][trials])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
stimColors = 'rmbc'
for i,(stim,clr,lbl) in enumerate(zip(stimNames,stimColors,('VIS+','VIS-','AUD+','AUD-'))):
    ax.plot(np.concatenate(x[i]),np.concatenate(y[i]),'o',mec=clr,mfc='none',alpha=0.5,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlabel('Latent '+str(lat+1)+' state',fontsize=14)
ax.set_ylabel('Prob resp',fontsize=14)
ax.legend()
plt.tight_layout()


# plot choice rule for two latents
lat = [1,0]
latInd = latentOrder['disrnn'][latPenInd][updPenInd][lat]
binSize = 0.1
bins = np.arange(-2,2+binSize,binSize)
z = [[] for _ in stimNames]
n = copy.deepcopy(z)
for obj,state,pr in zip(np.array(sessionData)[testIndex],latentStates['disrnn'][latPenInd][updPenInd],probResp['disrnn'][latPenInd][updPenInd]):
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
        for sessionInd,obj in enumerate(np.array(sessionData)[testIndex]):
            y.append([])
            if src == 'mice':
                resp = obj.trialResponse
            elif src == 'model prediction':
                resp = probResp[modelType][latPenInd][updPenInd][sessionInd][:obj.nTrials]
            elif src == 'model simulation':
                resp = simResp[modelType][latPenInd][updPenInd][sessionInd][:obj.nTrials]
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
            trialResponse = [simResp[modelType][latPenInd][updPenInd][sessionInd]]
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





