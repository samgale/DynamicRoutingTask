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
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getRNNSessions,getSessionData

from absl import app
from absl import flags
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import plotting
from disentangled_rnns.library import rnn_utils
import optax


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"


# define dataset
summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
isStandardRegimen = getIsStandardRegimen(summaryDf)
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])

maxTrainSessions = 20
mouseIds = []
sessionStartTimes = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = getRNNSessions(mouseId,df)
    if len(sessions) > maxTrainSessions:
        mouseIds.append(mouseId)
        sessionStartTimes.append([st.strftime('%Y%m%d_%H%M%S') for st in df.loc[sessions,'start time']])

sessionData = [getSessionData(mouseIds[0],st) for st in sessionStartTimes[0]][:22]
maxTrials = max(session.nTrials for session in sessionData)
nInputs = 6
stimNames = ['vis1','vis2','sound1','sound2']

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
    
testInd = [2,3]
trainInd = np.setdiff1d(np.arange(len(sessionData)),testInd)    
testDataset,trainDataset = [rnn_utils.DatasetRNN(
        xs=modelInput[:,i],
        ys=targetOutput[:,i],
        y_type='categorical',
        n_classes=2,
        x_names=stimNames+['prev resp','prev outcome'],
        y_names=['resp'],
        batch_size=1,
        batch_mode='random')
    for i in (testInd,trainInd)]


#
# FLAGS = flags.FLAGS
# flags.DEFINE_integer("n_steps_per_session", maxTrials, "Number of steps per session in the dataset.")
# flags.DEFINE_integer("n_sessions", len(sessionData), "Number of sessions in the dataset.")
# flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate.")
# flags.DEFINE_integer("n_warmup_steps", 1000, "Number of training warmup steps.")
# flags.DEFINE_integer("n_training_steps", 3000, "Number of main training steps.")


# define the disRNN architecture
disrnn_config = disrnn.DisRnnConfig(
    # Dataset related
    obs_size=nInputs,
    output_size=2,
    x_names=testDataset.x_names,
    y_names=testDataset.y_names,
    # Network architecture
    latent_size=5,
    update_net_n_units_per_layer=16,
    update_net_n_layers=4,
    choice_net_n_units_per_layer=4,
    choice_net_n_layers=2,
    activation="leaky_relu",
    # Penalties
    noiseless_mode=False,
    latent_penalty=0.001,
    update_net_obs_penalty=0.01,
    update_net_latent_penalty=0.01,
    choice_net_latent_penalty=0.001,
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
    do_plot=True)

# Additional training using information penalty
params, _, _ = rnn_utils.train_network(
    make_disrnn,
    training_dataset=trainDataset,
    validation_dataset=testDataset,
    loss="penalized_categorical",
    params=params,
    opt_state=None,
    opt=opt,
    n_steps=8000,
    do_plot=True)


# Plot bottleneck structure and update rules
plotting.plot_bottlenecks(params, disrnn_config)
plotting.plot_update_rules(params, disrnn_config)

# Eval disRNN on unseen data #
# Use the wamrup disrnn, so that there will be no noise
i = 0 # session index
xs = testDataset._xs[:,[i]]
ys = testDataset._ys[:,[i]]
network_outputs,network_states = rnn_utils.eval_network(make_disrnn_warmup, params, xs)

likelihood = rnn_utils.normalized_likelihood(ys,network_outputs[:,[i],:2])

probResp = np.exp(network_outputs[:,i,1]) / (np.exp(network_outputs[:,i,0]) + np.exp(network_outputs[:,i,1]))
















