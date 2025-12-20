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
from DynamicRoutingAnalysisUtils import getFirstExperimentSession,getSessionsToPass,getSessionData


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


def getModelInputAndTarget(session,inputSize,isFitToMouse,device):
    modelInput = np.zeros((session.nTrials,inputSize),dtype=np.float32)
    for i,stim in enumerate(('vis1','vis2','sound1','sound2')):    
        modelInput[:,i] = session.trialStim == stim
    if isFitToMouse:
        modelInput[1:,4] = session.trialResponse[:-1]
        modelInput[1:,5] = session.trialRewarded[:-1]
    modelInput = torch.from_numpy(modelInput).to(device)
    targetOutput = torch.from_numpy((session.trialResponse if isFitToMouse else session.trialStim==session.rewardedStim).astype(np.float32)).to(device)
    return modelInput,targetOutput


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
    

sessionData = [getSessionData(mice[-1],startTime,lightLoad=True) for startTime in sessions[-1]]
testData = [sessionData[0]]
trainData = [session for session in sessionData if session not in testData]


sessionData = [[getSessionData(m,st) for st in random.sample(list(s),2)] for m,s in zip(mice,sessions)]
trainData,testData = [[s[i] for s in sessionData] for i in (0,1)]
assert(not any(np.isin(trainData,testData)))



isFitToMouse = True
isSimulation = not isFitToMouse

inputSize = 6
hiddenSize = 50
outputSize = 1
dropoutProb = 0
learningRate = 0.001 # 0.001
smoothingConstants = (0.9,0.999) # (0.9,0.999)
weightDecay = 0.01 # 0.01

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
model = CustomLSTM(inputSize,hiddenSize,outputSize,dropoutProb).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=learningRate,betas=smoothingConstants,weight_decay=weightDecay)
lossFunc = torch.nn.BCELoss()

trainingIter = 0
logLossTrain = []
logLossTest = []

nTrainIters = 6000
for _ in range(nTrainIters):
    trainingIter += 1
    print('training iter '+str(trainingIter))
    logLossTrain.append([])
    logLossTest.append([])
    random.shuffle(trainData)
    model.train()
    for session in random.sample(trainData,1):
        modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
        modelOutput = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
        loss = lossFunc(modelOutput,targetOutput)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        logLossTrain[-1].append(loss.item())
    model.eval()
    with torch.no_grad():
        for session in random.sample(testData,1):
            modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
            modelOutput = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
            logLossTest[-1].append(lossFunc(modelOutput,targetOutput).item())


smoothSamples = 30
smoothFilter = np.ones(smoothSamples) / smoothSamples

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# ax.plot(np.median(logLossTrain,axis=(1)),'r',label='training')
# ax.plot(np.median(logLossTest,axis=(1)),'b',label='testing')
ax.plot(np.convolve(np.median(logLossTrain,axis=(1)),smoothFilter,mode='same'),'r',label='training')
ax.plot(np.convolve(np.median(logLossTest,axis=(1)),smoothFilter,mode='same'),'b',label='testing')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,0.8])
ax.set_xlabel('training iteration')
ax.set_ylabel('-log(likelihood)')
ax.legend()
plt.tight_layout()

# filePath = os.path.join(baseDir,'Sam','RNNmodel','model_weights.pth')
# torch.save(model.state_dict(),filePath)

# model = CustomLSTM(inputSize,hiddenSize,outputSize,dropoutProb)
# model.load_state_dict(torch.load(filePath,weights_only=True))

pAction = []
action = []
reward = []
model.eval()
with torch.no_grad():
    for session in testData[:5]:
        modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
        pAct,act,rew = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation=True)
        pAction.append(pAct)
        action.append(act)
        reward.append(rew)
        

for session,pAct,act in zip(testData,pAction,action):
    for i,rewStim in enumerate(session.blockStimRewarded):
        for stim in ('vis1','sound1','vis2','sound2'):
            print(rewStim,stim,pAct[(session.trialBlock==i+1) & (session.trialStim==stim)].mean())











