# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 12:40:00 2026

@author: svc_ccg
"""

import copy
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getSessionsToPass,getRNNSessions,getSessionData

from absl import app
from absl import flags
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import plotting
from disentangled_rnns.library import rnn_utils
import optax


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"


# get data for pooled training
summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
isStandardRegimen = getIsStandardRegimen(summaryDf)
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])

sessionData = {phase: [] for phase in ('initial training','after learning')}
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
    sessions = np.where(sessions)[0]
    sessionsToPass = getSessionsToPass(mouseId,df,sessions,stage=5)
    sessionData['initial training'].append([getSessionData(mouseId,startTime,lightLoad=True) for startTime in df.loc[sessions[:2],'start time']])
    sessionData['after learning'].append([getSessionData(mouseId,startTime,lightLoad=True) for startTime in df.loc[sessions[sessionsToPass:sessionsToPass+2],'start time']])
    
    

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
            batch_size=1,
            batch_mode='random')
        for i in (testIndex,trainIndex)]
    return testDataset,trainDataset
    

trainingPhase = 'initial training'
# first session from odd mice, second session from even mice
dataSplit1 = [d[0] for d in sessionData[trainingPhase][::2]] + [d[1] for d in sessionData[trainingPhase][1::2]]
# second session from odd mice, first session from even mice
dataSplit2 = [d[1] for d in sessionData[trainingPhase][::2]] + [d[0] for d in sessionData[trainingPhase][1::2]]
testIndex = np.arange(len(dataSplit1))
trainIndex = np.arange(len(mice),2*len(mice))
testDataset,trainDataset = getDisrnnDataset(dataSplit1+dataSplit2,testIndex)

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
    latent_penalty=0.01,
    update_net_obs_penalty=0.01,
    update_net_latent_penalty=0.01,
    choice_net_latent_penalty=0.01,
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
    n_steps=9000,
    do_plot=True)
    



# Plot bottleneck structure and update rules
plotting.plot_bottlenecks(params, disrnn_config)
# plotting.plot_choice_rule(params, disrnn_config)
# plotting.plot_update_rules(params, disrnn_config)

# Eval disRNN on unseen data #
# Use the wamrup disrnn, so that there will be no noise
latentStates = []
probResp = []
likelihood = []
for i in np.arange(len(testIndex)):
    xs = testDataset._xs[:,[i]]
    ys = testDataset._ys[:,[i]]
    network_outputs,network_states = rnn_utils.eval_network(make_disrnn_warmup,params,xs)
    latentStates.append(network_states[:,0])
    probResp.append(np.exp(network_outputs[:,0,1]) / (np.exp(network_outputs[:,0,0]) + np.exp(network_outputs[:,0,1])))
    likelihood.append(rnn_utils.normalized_likelihood(ys,network_outputs[:,0,:2]))





#
param_prefix = 'hk_disentangled_rnn'
latent_sigmas = np.array(disrnn.reparameterize_sigma(params[param_prefix]['latent_sigma_params']))
latent_order = np.argsort(latent_sigmas)


resp = xs[:,0,4].copy()
resp[resp<1] = np.nan
rew = xs[:,0,5].copy()
rew[rew<1] = np.nan
for i in latent_order:
    plt.figure()
    plt.plot(network_states[:,0,i])
    plt.plot(resp,'bo')
    plt.plot(rew,'ro',ms=4)
    

#
for ind in latent_order:
    fig = plt.figure()
    state = network_states[:,0,ind]
    for r,rewStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,r+1)
        deltaState = np.zeros((4,2))
        blockTypeTrials = obj.rewardedStim==rewStim
        for i,stim in enumerate(stimNames):
            stimTrials = np.where(blockTypeTrials & (obj.trialStim==stim))[0]
            stimTrials = stimTrials[stimTrials > 0]
            for j,resp in enumerate((1,0)):
                trials = stimTrials[obj.trialResponse[stimTrials-1]==resp]
                deltaState[i,j] = np.mean(state[trials] - state[trials-1])
        cmax = np.max(np.absolute(deltaState))
        im = ax.imshow(deltaState,cmap='bwr',clim=(-cmax,cmax))
        cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
        
            
        








