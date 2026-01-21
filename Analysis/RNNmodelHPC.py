# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import argparse
import copy
import os
import pathlib
import random
import numpy as np
import pandas as pd
import torch
from  DynamicRoutingAnalysisUtils import getSessionsToPass, getSessionData, getPerformanceStats


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting')


class CustomRNN(torch.nn.Module):
    def __init__(self,hiddenType,inputSize,hiddenSize,outputSize,dropoutProb):
        super(CustomRNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.hiddenType = hiddenType
        if hiddenType == 'lstm':
            self.hidden = torch.nn.LSTMCell(inputSize,hiddenSize,bias=True)
        elif hiddenType == 'gru':
            self.hidden = torch.nn.GRUCell(inputSize,hiddenSize,bias=True)
        else:
            self.hidden = torch.nn.RNNCell(inputSize,hiddenSize,bias=True)
        self.dropout = torch.nn.Dropout(dropoutProb)
        self.linear = torch.nn.Linear(hiddenSize,outputSize)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,inputSequence,trialStim=None,rewardedStim=None,rewardScheduled=None,isSimulation=False):
        hiddenState = torch.zeros(self.hiddenSize).to(inputSequence.device)
        if self.hiddenType == 'lstm':
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
            
            if self.hiddenType == 'lstm':    
                hiddenState,cellState = self.hidden(currentInput,(hiddenState,cellState))
            else:
                hiddenState = self.hidden(currentInput,hiddenState)
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


def trainModel(testData,trainData,hiddenType,nTrainSessions,nHiddenUnits):
    isFitToMouse = True
    isSimulation = not isFitToMouse
    inputSize = 6
    hiddenSize = nHiddenUnits
    outputSize = 1
    dropoutProb = 0
    learningRate = 0.001 # 0.001
    smoothingConstants = (0.9,0.999) # (0.9,0.999)
    weightDecay = 0.01 # 0.01
    maxTrainIters = 30000
    earlyStopThresh = 0.05
    earlyStopIters = 500
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    lossFunc = torch.nn.BCELoss()

    model = CustomRNN(hiddenType,inputSize,hiddenSize,outputSize,dropoutProb).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=learningRate,betas=smoothingConstants,weight_decay=weightDecay)
    logLossTrain = np.full(maxTrainIters,np.nan)
    logLossTest = np.full(maxTrainIters,np.nan)
    bestIter = 0
    trainIndex = 0
    for i in range(maxTrainIters):
        session = trainData[trainIndex]
        if trainIndex == nTrainSessions - 1:
            random.shuffle(trainData)
            trainIndex = 0
        else:
            trainIndex += 1
        model.train()
        modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
        modelOutput = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
        loss = lossFunc(modelOutput,targetOutput)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        logLossTrain[i] = loss.item()
        
        model.eval()
        with torch.no_grad():
            session = testData
            modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
            modelOutput = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
            logLossTest[i] = lossFunc(modelOutput,targetOutput).item()
            if logLossTest[i] < logLossTest[bestIter]:
                bestIter = i
                bestModelStateDict = copy.deepcopy(model.state_dict())
                
        if i > bestIter + earlyStopIters and np.all(logLossTest[i-earlyStopIters:i+1] > logLossTest[bestIter] + earlyStopThresh):
            break
    
    model.load_state_dict(bestModelStateDict)
    model.eval()
    with torch.no_grad():
        prediction = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0].cpu().numpy()
        simulation = []
        simAction = []
        for _ in range(10):
            pAction,action,reward = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation=True)
            simulation.append(pAction.cpu().numpy())
            simAction.append(action.cpu().numpy())

    fileName = testData.subjectName+'_'+testData.startTime+'_'+hiddenType+'_'+str(nTrainSessions)+'trainSessions_'+str(nHiddenUnits)+'hiddenUnits'+'.npz'
    filePath = os.path.join(baseDir,'Sam','RNNmodel',fileName)
    np.savez_compressed(filePath,testSession=testData.startTime,trainSessions=[session.startTime for session in trainData],
                        logLossTrain=logLossTrain[:i+1],logLossTest=logLossTest[:i+1],prediction=prediction,simulation=simulation,simAction=simAction) 


def getRNNSessions(mouseId,df):
    standardSessions = np.array(['stage 5' in task and not any(variant in task for variant in ('nogo','noAR','oneReward','rewardOnly','catchOnly')) for task in df['task version']]) & ~np.array(df['ignore']).astype(bool) & ~np.array(df['muscimol']).astype(bool)
    standardSessions = np.where(standardSessions)[0]
    sessionsToPass = getSessionsToPass(mouseId,df,standardSessions,stage=5)
    sessions = standardSessions[sessionsToPass-2:]
    hits,dprimeSame,dprimeOther = getPerformanceStats(df,sessions)
    sessions = sessions[np.sum(np.array(hits) > 9,axis=1) > 3]
    return sessions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=str)
    parser.add_argument('--maxTrainSessions',type=int)
    parser.add_argument('--nProcesses',type=int)
    args = parser.parse_args()
    mouseId = args.mouseId
    maxTrainSessions = args.maxTrainSessions
    nProcesses = args.nProcesses

    drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = getRNNSessions(mouseId,df)[:maxTrainSessions+1]
    sessionData = [getSessionData(mouseId,st) for st in df.loc[sessions,'start time']]

    torch.cuda.set_per_process_memory_fraction(1/nProcesses)
    torch.multiprocessing.set_start_method('spawn',force=True)

    poolArgs = []
    for testData in sessionData[2:4]:
        for hiddenType in ('rnn','gru','lstm'):
            for nTrainSessions in (4,8,12,16,20):
                for nHiddenUnits in (2,4,8,16,32):
                    trainData = [session for session in sessionData[:nTrainSessions+1] if session is not testData]
                    poolArgs.append((testData,trainData,hiddenType,nTrainSessions,nHiddenUnits))

    with torch.multiprocessing.Pool(processes=nProcesses) as pool:
        pool.starmap(trainModel,poolArgs)
    