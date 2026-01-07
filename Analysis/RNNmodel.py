# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:38:41 2025

@author: svc_ccg
"""

import copy
import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getFirstExperimentSession,getSessionsToPass,getSessionData


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


isStandardRegimen = getIsStandardRegimen(summaryDf)
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])
sessions = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    standardSessions = np.array(['stage 5' in task and not any(variant in task for variant in ('nogo','noAR','oneReward','rewardOnly','catchOnly')) for task in df['task version']]) & ~np.array(df['ignore']).astype(bool)
    standardSessions = np.where(standardSessions)[0]
    sessionsToPass = getSessionsToPass(mouseId,df,standardSessions,stage=5)
    sessions.append(df.loc[standardSessions,'start time'][sessionsToPass-2:])
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
dsort = np.sort([len(s) for s in sessions])
cumProb = np.array([np.sum(dsort>=i)/dsort.size for i in dsort])
ax.plot(dsort,cumProb,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,dsort[-1]+1])
ax.set_ylim([0,1.01])
ax.set_xlabel('# sessions')
ax.set_ylabel('Cumalative fraction of mice',fontsize=16)
plt.tight_layout()
    

sessionData = [[getSessionData(m,st) for st in random.sample(list(s),2)] for m,s in zip(mice,sessions)]


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
lossFunc = torch.nn.BCELoss()

nSessions = (1,5,10,20,40,80)
nModels = 10
nTrainIters = 6000
logLossTrain = [[[] for _ in range(nModels)] for _ in nSessions]
logLossTest = copy.deepcopy(logLossTrain)

for k,n in enumerate(nSessions):
    for m in range(nModels):
        if n == len(sessionData):
            trainData,testData = [[s[i] for s in sessionData] for i in (0,1)]
        else:
            trainData,testData = [[s[i] for s in random.sample(sessionData,n)] for i in (0,1)]
        model = CustomLSTM(inputSize,hiddenSize,outputSize,dropoutProb).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=learningRate,betas=smoothingConstants,weight_decay=weightDecay)
        for i in range(nTrainIters):
            if i % 100 == 0:
                print(str(n)+' sessions, model '+str(m+1)+', training iter '+str(i+1))
            
            model.train()
            session = random.choice(trainData)
            modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
            modelOutput = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
            loss = lossFunc(modelOutput,targetOutput)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            logLossTrain[k][m].append(loss.item())
            
            model.eval()
            with torch.no_grad():
                session = random.choice(testData)
                modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
                modelOutput = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
                logLossTest[k][m].append(lossFunc(modelOutput,targetOutput).item())


def boxcar(data,smoothSamples):
    smoothFilter = np.ones(smoothSamples) / smoothSamples
    smoothedData = np.convolve(data,smoothFilter,mode='same')
    smoothedData[:smoothSamples] = smoothedData[smoothSamples]
    smoothedData[-smoothSamples:] = smoothedData[-smoothSamples]
    return smoothedData


for k,n in enumerate(nSessions):
    fig = plt.figure(figsize=(6,8))
    gs = matplotlib.gridspec.GridSpec(5,2)
    for m in range(nModels):
        i = m if m < 5 else m - 5
        j = 0 if m < 5 else 1
        ax = fig.add_subplot(gs[i,j])
        ax.plot(boxcar(logLossTrain[k][m],n),'r',label='train')
        ax.plot(boxcar(logLossTest[k][m],n),'b',label='test')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim([0,1])
        if i==4 and j==0:
            ax.set_xlabel('training iteration')
        if i==2 and j==0:
            ax.set_ylabel('-log(likelihood)')
        if i==0 and j==1:
            ax.legend()
    plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for k,n in enumerate(nSessions):
    for loss,clr,lbl in zip((logLossTrain[k],logLossTest[k]),'rb',('train','test')):
        smoothedLoss = [boxcar(s,n) for s in loss]
        d = [min(s) for s in smoothedLoss]
        m = np.mean(d)
        s = np.std(d) / (len(d)**0.5)
        ax.plot(n,m,'o',mfc=clr,mec=clr,label=(lbl if k==0 else None))
        ax.plot([n,n],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('number of unique train/test sessions')
ax.set_ylabel('-log(likelihood)')
ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for k,n in enumerate(nSessions):
    for loss,clr,lbl in zip((logLossTrain[k],logLossTest[k]),'rb',('train','test')):
        smoothedLoss = [boxcar(s,n) for s in loss]
        d = [np.exp(-min(s)) for s in smoothedLoss]
        m = np.mean(d)
        s = np.std(d) / (len(d)**0.5)
        ax.plot(n,m,'o',mfc=clr,mec=clr,label=(lbl if k==0 else None))
        ax.plot([n,n],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('number of unique train/test sessions')
ax.set_ylabel('likelihood')
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











