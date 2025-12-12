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
    def __init__(self,inputSize,hiddenSize,outputSize):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTMCell(inputSize,hiddenSize,bias=True)
        self.linear = nn.Linear(hiddenSize,outputSize)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inputSequence,isSimulation=False,trialStim=None,rewardedStim=None,rewardScheduled=None):
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
                
            h_t,c_t = self.lstm(currentInput)
            output = self.linear(h_t)
            pAction.append(self.sigmoid(output)[0])
            if isSimulation:
                action.append(random.random() < pAction[-1])
                reward.append((action[-1] and trialStim[t] == rewardedStim[t]) or rewardScheduled[t])
        
        return torch.stack(pAction),torch.tensor(action),torch.tensor(reward)


def getTrialSamples(nTrials,minTrials=20,maxTrials=40):
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


filePath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data\818720\DynamicRouting1_818720_20251202_150802.hdf5"
sessionData = DynRoutData()
sessionData.loadBehavData(filePath,lightLoad=True)
nTrials = sessionData.nTrials

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
learningRate = 0.01 
smoothingConstant = 0.99

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
models = [[CustomLSTM(inputSize,hiddenSize,outputSize).to(device) for _ in range(cvFolds)] for _ in range(cvIters)]
optimizers = [[torch.optim.RMSprop(models[i][j].parameters(),lr=learningRate,alpha=smoothingConstant) for j in range(cvFolds)] for i in range(cvIters)]
lossFunc = nn.BCELoss()

trainingIter = 0
shuffleInd = [np.random.permutation(nTrials) for _ in range(cvIters)]
logLossTrain = [[[] for _ in range(cvFolds)] for _ in range(cvIters)]
logLossTest = [[[] for _ in range(cvFolds)] for _ in range(cvIters)]

modelInput = np.zeros((nTrials,inputSize),dtype=np.float32)
for i,stim in enumerate(('vis1','vis2','sound1','sound2')):    
    modelInput[:,i] = sessionData.trialStim == stim
if isFitToMouse:
    modelInput[1:,4] = sessionData.trialResponse[:-1]
    modelInput[1:,5] = sessionData.trialRewarded[:-1]
modelInput = torch.from_numpy(modelInput).to(device)

targetOutput = torch.from_numpy((sessionData.trialResponse if isFitToMouse else sessionData.trialStim==sessionData.rewardedStim).astype(np.float32)).to(device)


nTrainIters = 51
for _ in range(nTrainIters):
    trainingIter += 1
    print('training iter '+str(trainingIter))
    for i in range(cvIters):
        for j in range(cvFolds):
            start = j * nTestTrials
            testTrials = shuffleInd[i][start:start+nTestTrials] if j+1 < cvFolds else shuffleInd[i][start:]
            trainTrials = np.setdiff1d(shuffleInd[i],testTrials)
            logLossTrain[i][j].append([])
            logLossTest[i][j].append([])
            for sample in getTrialSamples(nTrials):
                lossTrials = np.isin(sample,trainTrials)
                evalTrials = np.isin(sample,testTrials)
                if np.any(lossTrials) and np.any(evalTrials):
                    modelOutput = models[i][j](modelInput[sample],isSimulation,trialStim[sample],rewardedStim[sample],rewardScheduled[sample])[0]
                    loss = lossFunc(modelOutput[lossTrials],targetOutput[sample][lossTrials])
                    loss.backward()
                    optimizers[i][j].step()
                    optimizers[i][j].zero_grad()
                    logLossTrain[i][j][-1].append(loss.item())
                    logLossTest[i][j][-1].append(lossFunc(modelOutput[evalTrials],targetOutput[sample][evalTrials]).item())


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.median(logLossTrain,axis=(0,1,3)),'r',label='training')
ax.plot(np.median(logLossTest,axis=(0,1,3)),'b',label='testing')
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
            pAct,act,rew = models[i][j](modelInput,isSimulation=True,trialStim=trialStim,rewardedStim=rewardedStim,rewardScheduled=rewardScheduled)
            pAction.append(pAct)
            action.append(act)
            reward.append(rew)
        

for i,rewStim in enumerate(sessionData.blockStimRewarded):
    for stim in ('vis1','sound1','vis2','sound2'):
        print(rewStim,stim,pAct[(sessionData.trialBlock==i+1) & (sessionData.trialStim==stim)].mean())











