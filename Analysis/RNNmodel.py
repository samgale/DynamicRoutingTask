# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:38:41 2025

@author: svc_ccg
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData,getFirstExperimentSession,getSessionsToPass,getSessionData


class CustomLSTM(torch.nn.Module):
    def __init__(self,inputSize,hiddenSize,outputSize,dropoutProb):
        super(CustomLSTM, self).__init__()
        self.hiddenSize = hiddenSize
        self.lstm = torch.nn.LSTMCell(inputSize,hiddenSize,bias=True)
        self.dropout = torch.nn.Dropout(dropoutProb)
        self.linear = torch.nn.Linear(hiddenSize,outputSize)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,inputSequence,trialStim=None,rewardedStim=None,rewardScheduled=None,isSimulation=False):
        hiddenState = torch.zeros(self.hiddenSize).to(inputSequence.device)
        cellState = torch.zeros(self.hiddenSize).to(inputSequence.device)
        pAction = []
        action = []
        reward = []
        for t in range(inputSequence.size(0)):
            if isSimulation and t > 0:
                currentInput = inputSequence[t].clone()
                currentInput[4] = action[-1]
                currentInput[5] = float(reward[-1])
            else:
                currentInput = inputSequence[t]
                
            hiddenState,cellState = self.lstm(currentInput,(hiddenState,cellState))
            output = self.dropout(hiddenState)
            output = self.linear(output)
            output = self.sigmoid(output)
            pAction.append(output[0])
            if isSimulation:
                action.append(random.random() < pAction[-1])
                reward.append((action[-1] and trialStim[t] == rewardedStim[t]) or rewardScheduled[t])
        
        return torch.stack(pAction),torch.tensor(action),torch.tensor(reward)


def getTrialSamples(nTrials,minTrials=50,maxTrials=100):
    trials = np.arange(nTrials)
    samples = []
    start = 0
    while True:
        end = start + random.randint(minTrials,maxTrials)
        samples.append(trials[start:end])
        start += int((end - start) / 2)
        if start >= nTrials:
            break
        elif nTrials - start < minTrials:
            samples[-1] = np.append(samples[-1],np.arange(start,nTrials))
            break
    random.shuffle(samples)
    return samples


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]

nSessions = 5

hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])
ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['cannula'] & ~summaryDf['stage 5 repeats']
mice = np.array(summaryDf[ind]['mouse id'])
sessions = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    preExperimentSessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore']).astype(bool)
    firstExperimentSession = getFirstExperimentSession(df)
    if firstExperimentSession is not None:
        preExperimentSessions[firstExperimentSession:] = False
    preExperimentSessions = np.where(preExperimentSessions)[0]
    sessionsToPass = getSessionsToPass(mouseId,df,preExperimentSessions,stage=5)
    sessions.append(df.loc[preExperimentSessions,'start time'][sessionsToPass:sessionsToPass+nSessions])
    

sessionData = [getSessionData(mice[0],startTime,lightLoad=True) for startTime in sessions[0]]
testData = sessionData[0]
trainData = sessionData[1:]


filePath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data\823257\DynamicRouting1_823257_20251212_134943.hdf5"
sessionData = DynRoutData()
sessionData.loadBehavData(filePath,lightLoad=True)

isFitToMouse = True
isSimulation = not isFitToMouse
trialStim = sessionData.trialStim
rewardedStim =sessionData.rewardedStim
rewardScheduled = sessionData.autoRewardScheduled

cvIters = 5
cvFolds = 5
nTestTrials = round(nTrials / cvFolds)

inputSize = 6
hiddenSize = 50
outputSize = 1
dropoutProb = 0.2
learningRate = 0.001
smoothingConstants = (0.9,0.999)
weightDecay = 0.01

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
model = CustomLSTM(inputSize,hiddenSize,outputSize,dropoutProb).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=learningRate,betas=smoothingConstants,weight_decay=weightDecay)
lossFunc = torch.nn.BCELoss()

modelInput = np.zeros((nTrials,inputSize),dtype=np.float32)
for i,stim in enumerate(('vis1','vis2','sound1','sound2')):    
    modelInput[:,i] = sessionData.trialStim == stim
if isFitToMouse:
    modelInput[1:,4] = sessionData.trialResponse[:-1]
    modelInput[1:,5] = sessionData.trialRewarded[:-1]
modelInput = torch.from_numpy(modelInput).to(device)

targetOutput = torch.from_numpy((testData.trialResponse if isFitToMouse else testData.trialStim==testData.rewardedStim).astype(np.float32)).to(device)
prediction = torch.zeros(testData.nTrials,dtype=torch.float32,requires_grad=False).to(device)

trainingIter = 0
shuffleInd = [np.random.permutation(nTrials) for _ in range(cvIters)]
logLossTrain = [[[] for _ in range(cvFolds)] for _ in range(cvIters)]
logLossTest = [[] for _ in range(cvIters)]

nTrainIters = 31
for _ in range(nTrainIters):
    trainingIter += 1
    print('training iter '+str(trainingIter))
    model.train()
    for session in getTrialSamples(nTrials):
        lossTrials = np.isin(sample,trainTrials)
        if np.any(lossTrials):
            modelOutput = models[i][j](modelInput[sample],trialStim[sample],rewardedStim[sample],rewardScheduled[sample],isSimulation)[0]
            loss = lossFunc(modelOutput[lossTrials],targetOutput[sample][lossTrials])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[i][j].parameters(),max_norm=1.0)
            optimizers[i][j].step()
            optimizers[i][j].zero_grad()
    models[i][j].eval()
    with torch.no_grad():
        modelOutput = models[i][j](modelInput,trialStim,rewardedStim,rewardScheduled,isSimulation)[0]
        logLossTrain[i][j].append(lossFunc(modelOutput[trainTrials],targetOutput[trainTrials]).item())
        prediction[testTrials] = modelOutput[testTrials].detach()
logLossTest[i].append(lossFunc(prediction,targetOutput).item())


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.mean(logLossTrain,axis=(0,1)),'r',label='training')
ax.plot(np.mean(logLossTest,axis=(0)),'b',label='testing')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('training iteration')
ax.set_ylabel('-log(likelihood)')
ax.legend()
plt.tight_layout()


pAction = []
action = []
reward = []
for i in range(cvIters):
    for j in range(cvFolds):
        models[i][j].eval()
        with torch.no_grad():
            pAct,act,rew = models[i][j](modelInput,trialStim,rewardedStim,rewardScheduled,isSimulation=True)
            pAction.append(pAct)
            action.append(act)
            reward.append(rew)
        

for i,rewStim in enumerate(sessionData.blockStimRewarded):
    for stim in ('vis1','sound1','vis2','sound2'):
        print(rewStim,stim,pAct[(sessionData.trialBlock==i+1) & (sessionData.trialStim==stim)].mean())











