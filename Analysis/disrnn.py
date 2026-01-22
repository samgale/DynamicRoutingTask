# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 12:40:00 2026

@author: svc_ccg
"""

import copy
import os
import numpy as np
import pandas as pd
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

sessionData = [getSessionData(mouseIds[0],st) for st in sessionStartTimes[0]]
maxTrials = max(session.nTrials for session in sessionData)
nInputs = 6
stimNames = ('vis1','vis2','sound1','sound2')

modelInput = -1 * np.ones((maxTrials,len(sessionData),nInputs),dtype=np.float32)
targetOutput = -1 * np.ones((maxTrials,len(sessionData),1),dtype=int)
for i,session in enumerate(sessionData):
    n = session.nTrials
    for j,stim in enumerate(stimNames):    
        modelInput[:n,i,j] = session.trialStim == stim
    modelInput[0,i,4:6] = 0
    modelInput[1:n,i,4] = session.trialResponse[:-1]
    modelInput[1:n,i,5] = session.trialRewarded[:-1]
    targetOutput[:n,i,0] = session.trialResponse
    
testInd = [0]
trainInd = np.setdiff1d(np.arange(len(sessionData)),testInd)    
testDataset,trainDataset = [rnn_utils.DatasetRNN(xs=modelInput[:,i],
                                                 ys=targetOutput[:,i],
                                                 y_type='categorical',
                                                 n_classes=2,
                                                 x_names=stimNames+('prev resp','prev outcome'),
                                                 y_names=('resp',),
                                                 batch_size=1,
                                                 batch_mode='random')
                            for i in (testInd,trainInd)]


# define the disRNN architecture
disrnn_config = disrnn.DisRnnConfig(
    # Dataset related
    obs_size=2,
    output_size=2,
    x_names=dataset.x_names,
    y_names=dataset.y_names,
    # Network architecture
    latent_size=5,
    update_net_n_units_per_layer=8,
    update_net_n_layers=4,
    choice_net_n_units_per_layer=4,
    choice_net_n_layers=2,
    activation="leaky_relu",
    # Penalties
    noiseless_mode=False,
    latent_penalty=1e-2,
    update_net_obs_penalty=1e-3,
    update_net_latent_penalty=1e-3,
    choice_net_latent_penalty=1e-3,
    l2_scale=1e-5,
)


































