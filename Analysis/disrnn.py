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
      
    def step(self, attempted_choice: int, trial_index: int) -> tuple[int, float | int, int]:
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
    
    

def getDisrnnDataset(sessionData,testIndex):
    nInputs = 6
    stimNames = ['vis1','vis2','sound1','sound2']
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
            x_names=stimNames+['prev resp','prev outcome'],
            y_names=['resp'],
            batch_size=1024,
            batch_mode='random') # random or rolling
        for i in (testIndex,trainIndex)]
    return testDataset,trainDataset
    

trainingPhase = 'after learning'
sessionData = (# first session from odd mice, second session from even mice
               [d[0] for d in sessionDataByMouse[trainingPhase][::2]] + [d[1] for d in sessionDataByMouse[trainingPhase][1::2]]
               # second session from odd mice, first session from even mice
               + [d[1] for d in sessionDataByMouse[trainingPhase][::2]] + [d[0] for d in sessionDataByMouse[trainingPhase][1::2]])
testIndex = np.arange(len(mice))
trainIndex = np.arange(len(mice),2*len(mice))
testDataset,trainDataset = getDisrnnDataset(sessionData,testIndex)


latentPenalties = [0.01,0.005,0.001,0.0005,0.0001]
updatePenalties = [0.01,0.007,0.005,0.003,0.001]
modelParams = [[] for _ in range(len(latentPenalties))]
modelConfig = copy.deepcopy(modelParams)
latentSigmas = copy.deepcopy(modelParams)
latentOrder = copy.deepcopy(modelParams)
latentStates = [[[] for _ in range(len(updatePenalties))] for _ in range(len(latentPenalties))]
probResp = copy.deepcopy(latentStates)
likelihood = copy.deepcopy(latentStates)
for i,latPen in enumerate(latentPenalties):
    for j,updPen in enumerate(updatePenalties):
        # define the disRNN architecture
        disrnn_config = disrnn.DisRnnConfig(
            # Dataset related
            obs_size=6,
            output_size=2,
            x_names=testDataset.x_names,
            y_names=testDataset.y_names,
            # Network architecture
            latent_size=9,
            update_net_n_units_per_layer=16,
            update_net_n_layers=8,
            choice_net_n_units_per_layer=4,
            choice_net_n_layers=2,
            activation="leaky_relu",
            # Penalties
            noiseless_mode=False,
            latent_penalty=latPen,
            update_net_obs_penalty=updPen,
            update_net_latent_penalty=updPen,
            choice_net_latent_penalty=latPen,
            l2_scale=1e-5)
        
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
        
        # Define an optimizer
        opt = optax.adam(learning_rate=0.001)
        
        # Warmup training with no noise and no penalties
        params, _, _ = rnn_utils.train_network(
            make_disrnn_warmup,
            training_dataset=trainDataset,
            validation_dataset=testDataset,
            loss="penalized_categorical",
            params=None,
            opt_state=None,
            opt=opt,
            n_steps=1000,
            do_plot=False)
        
        # Additional training using information penalty
        params, _, _ = rnn_utils.train_network(
            make_disrnn,
            training_dataset=trainDataset,
            validation_dataset=testDataset,
            loss="penalized_categorical",
            params=params,
            opt_state=None,
            opt=opt,
            n_steps=10000,
            do_plot=True)
        
        # store model params
        modelParams[i].append(params)
        modelConfig[i].append(disrnn_config)
        latentSigmas[i].append(np.array(disrnn.reparameterize_sigma(params['hk_disentangled_rnn']['latent_sigma_params'])))
        latentOrder[i].append(np.argsort(latentSigmas[i][-1]))
        
        # Eval disRNN on unseen data #
        # Use the wamrup disrnn, so that there will be no noise
        for s in np.arange(len(testIndex)):
            xs = testDataset._xs[:,[s]]
            ys = testDataset._ys[:,[s]]
            network_outputs,network_states = rnn_utils.eval_network(make_disrnn_warmup,params,xs)
            latentStates[i][j].append(network_states[:,0])
            probResp[i][j].append(np.exp(network_outputs[:,0,1]) / (np.exp(network_outputs[:,0,0]) + np.exp(network_outputs[:,0,1])))
            likelihood[i][j].append(rnn_utils.normalized_likelihood(ys,network_outputs[:,:,:2]))


# simulate behavior with trained networks
simResp = latentStates = [[[] for _ in range(len(updatePenalties))] for _ in range(len(latentPenalties))]
for i,latPen in enumerate(latentPenalties):
    for j,updPen in enumerate(updatePenalties):
        print(i,j)
        disrnn_config= copy.deepcopy(modelConfig[i][j])
        disrnn_config.latent_penalty = 0
        disrnn_config.choice_net_latent_penalty = 0
        disrnn_config.update_net_obs_penalty = 0
        disrnn_config.update_net_latent_penalty = 0
        disrnn_config.l2_scale = 0
        disrnn_config.noiseless_mode = True
        make_disrnn = lambda: disrnn.HkDisentangledRNN(disrnn_config)
        agent = two_armed_bandits.AgentNetwork(make_disrnn,modelParams[i][j])
        for sessionInd,session in enumerate(np.array(sessionData)[testIndex]):
            networkInput = testDataset._xs[:,[sessionInd]]
            env = DynamicRoutingEnvironment(networkInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled)
            d = two_armed_bandits.create_dataset(agent,env,session.nTrials,1)  
            simResp[i][j].append(d._ys[:,0,0])


#
likelihoodMat = np.zeros((len(latentPenalties),len(updatePenalties)))
for i,latPen in enumerate(latentPenalties):
    for j,updPen in enumerate(updatePenalties):
        likelihoodMat[i,j] = np.mean(likelihood[i][j])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(likelihoodMat,cmap='magma')
cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
# cb.set_ticks((0,0.2,0.4,0.6))
# cb.set_ticklabels((0,0.2,0.4,0.6),fontsize=12)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out')
ax.set_xticks(np.arange(len(updatePenalties)))
ax.set_yticks(np.arange(len(latentPenalties)))
ax.set_xticklabels(updatePenalties)
ax.set_yticklabels(latentPenalties)
ax.set_xlabel('Update penalty')
ax.set_ylabel('Latent penalty')
ax.set_title('Likelihood')
plt.tight_layout()


# Plot bottleneck structure and update rules
for i,latPen in enumerate(latentPenalties):
    for j,updPen in enumerate(updatePenalties):
        plotting.plot_bottlenecks(modelParams[i][j],modelConfig[i][j])

# plotting.plot_choice_rule(params, disrnn_config)
# plotting.plot_update_rules(params, disrnn_config)






#
# resp = xs[:,0,4].copy()
# resp[resp<1] = np.nan
# rew = xs[:,0,5].copy()
# rew[rew<1] = np.nan
# for i in latent_order:
#     plt.figure()
#     plt.plot(network_states[:,0,i])
#     plt.plot(resp,'bo')
#     plt.plot(rew,'ro',ms=4)
    

# 
latPenInd = 0
updPenInd = 0
for ind in latentOrder[latPenInd][updPenInd][:5]:
    fig = plt.figure()
    gs = gs = matplotlib.gridspec.GridSpec(2,2)
    for row,rewStim in enumerate(('vis1','sound1')):
        for col,resp in enumerate((1,0)):
            ax = fig.add_subplot(gs[row,col])
            deltaState = []
            for obj,state in zip(np.array(sessionData)[testIndex],latentStates[latPenInd][updPenInd]):
                state = state[:obj.nTrials,ind]
                ds = np.zeros((4,4))
                blockTypeTrials = obj.rewardedStim==rewStim
                for i,stim in enumerate(stimNames):
                    trials = np.where(blockTypeTrials & (obj.trialStim==stim) & ~obj.autoRewardScheduled)[0]
                    trials = trials[trials > 0]
                    for tr in trials:
                        if obj.trialResponse[tr-1]==resp:
                            prevStim = obj.trialStim[np.where(np.isin(obj.trialStim[:tr],stimNames))[0][-1]]
                            j = stimNames.index(prevStim)
                            ds[i,j] = np.mean(state[tr] - state[tr-1])
                deltaState.append(ds)
            deltaState = np.mean(deltaState,axis=0)
            cmax = np.max(np.absolute(deltaState))
            im = ax.imshow(deltaState,cmap='bwr',clim=(-cmax,cmax))
            cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
            
            
# block transition plot
preTrials = 5
postTrials = 20
x = np.arange(-preTrials,postTrials+1)
for src in ('mice','model'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.add_patch(matplotlib.patches.Rectangle([-0.5,0],width=5,height=1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for stimLbl,clr,ls in zip(('rewarded target stim','unrewarded target stim','non-target (rewarded modality)','non-target (unrewarded modality'),'gmgm',('-','-','--','--')):
        y = []
        for obj,pred in zip(np.array(sessionData)[testIndex],simResp[latPenInd][updPenInd]):
            y.append([])
            if src == 'mice':
                resp = obj.trialResponse
            else:
                resp = pred[:obj.nTrials]
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
            trialResponse = [simResp[latPenInd][updPenInd][sessionInd]]
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





