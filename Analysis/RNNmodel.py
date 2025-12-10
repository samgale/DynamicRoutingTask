# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:38:41 2025

@author: svc_ccg
"""

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData


class CustomLSTM(nn.Module):
    def __init__(self,inputSize,hiddenSize,outputSize,sessionData):
        super(CustomLSTM, self).__init__()
        self.sessionData = sessionData
        self.lstm = nn.LSTMCell(inputSize,hiddenSize,bias=True)
        self.linear = nn.Linear(hiddenSize,outputSize)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inputSequence,isSimulation=False):
        pAction = []
        action = []
        reward = []
        for t in range(self.sessionData.nTrials):
            if isSimulation and t > 0:
                inputSequence[t,4] = action[t-1]
                inputSequence[t,5] = float(reward[t-1])
                
            h_t,c_t = self.lstm(inputSequence[t])
            output = self.linear(h_t)
            pAction.append(self.sigmoid(output)[0])
            if isSimulation:
                action.append(random.random() < pAction[-1])
                reward.append((action[-1] and self.sessionData.trialStim[t] == self.sessionData.rewardedStim[t]) or self.sessionData.autoRewardScheduled[t])
            else:
                action.append(self.sessionData.trialResponse[t])
                reward.append(self.sessionData.trialRewarded[t])
        
        return torch.stack(pAction),np.array(action),np.array(reward)


filePath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data\818720\DynamicRouting1_818720_20251202_150802.hdf5"
sessionData = DynRoutData()
sessionData.loadBehavData(filePath,lightLoad=True)
nTrials = sessionData.nTrials


# model = CustomLSTM(inputSize,hiddenSize,outputSize,sessionData,isSimulation=False)
# pAction,action,reward = model(modelInput)
# pActionAsArray = pAction.detach().numpy()

cvIters = 5
cvFolds = 5
nTestTrials = round(nTrials / cvFolds)
shuffleInd = [np.random.permutation(nTrials) for _ in range(cvIters)]

inputSize = 6
hiddenSize = 50
outputSize = 1
learningRate = 0.01 
smoothingConstant = 0.99

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
models = [[CustomLSTM(inputSize,hiddenSize,outputSize,sessionData).to(device) for _ in range(cvFolds)] for _ in range(cvIters)]
optimizers = [[torch.optim.RMSprop(models[i][j].parameters(),lr=learningRate,alpha=smoothingConstant) for j in range(cvFolds)] for i in range(cvIters)]
lossFunc = nn.BCELoss()
logLossTrain = [[[] for _ in range(cvFolds)] for _ in range(cvIters)]
logLossTest = [[] for _ in range(cvIters)]
trainingIter = 0

modelInput = np.zeros((nTrials,inputSize),dtype=np.float32)
for i,stim in enumerate(('vis1','vis2','sound1','sound2')):    
    modelInput[:,i] = sessionData.trialStim == stim
modelInput[1:,4] = sessionData.trialResponse[:-1]
modelInput[1:,5] = sessionData.trialRewarded[:-1]
modelInput = torch.from_numpy(modelInput).to(device)

targetOutput = torch.from_numpy(sessionData.trialResponse.astype(np.float32)).to(device)

prediction = torch.zeros(nTrials,dtype=torch.float32).to(device)


nTrainIters = 100
for _ in range(nTrainIters):
    trainingIter += 1
    for i in range(cvIters):
        for j in range(cvFolds):
            print('training iter '+str(trainingIter)+', cv iter '+str(i+1)+', cv fold '+str(j+1))
            start = j * nTestTrials
            testTrials = shuffleInd[i][start:start+nTestTrials] if j+1 < cvFolds else shuffleInd[i][start:]
            trainTrials = np.setdiff1d(shuffleInd[i],testTrials)
            modelOutput = models[i][j](modelInput)[0]
            loss = lossFunc(modelOutput[trainTrials],targetOutput[trainTrials])
            loss.backward()
            optimizers[i][j].step()
            optimizers[i][j].zero_grad()
            logLossTrain[i][j].append(loss.item())
            prediction[testTrials] = modelOutput[testTrials]
        logLossTest[i].append(lossFunc(prediction,targetOutput).item())


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.median(logLossTrain,axis=(0,1)),'r',label='training')
ax.plot(np.median(logLossTest,axis=0),'b',label='testing')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('training iteration')
ax.set_ylabel('-log(liklihood)')
ax.legend()
plt.tight_layout()


pAction = []
action = []
reward = []
with torch.no_grad():
    for i in range(cvIters):
        for j in range(cvFolds):
            pAct,act,rew = models[i][j](modelInput,isSimulation=True)
            pAction.append(pAct)
            action.append(act)
            reward.append(rew)
        













