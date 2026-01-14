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
from  DynamicRoutingAnalysisUtils import getIsStandardRegimen, getSessionsToPass, getSessionData


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting')


class CustomRNN(torch.nn.Module):
    def __init__(self,inputSize,hiddenSize,outputSize,dropoutProb,hiddenType):
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


def trainModel(mouseId,nTrainSessions,nHiddenUnits,hiddenType):
    isFitToMouse = True
    isSimulation = not isFitToMouse
    inputSize = 6
    hiddenSize = nHiddenUnits
    outputSize = 1
    dropoutProb = 0
    learningRate = 0.001 # 0.001
    smoothingConstants = (0.9,0.999) # (0.9,0.999)
    weightDecay = 0.01 # 0.01
    maxTrainIters = 25000
    earlyStopThresh = 0.05
    earlyStopIters = 500
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    lossFunc = torch.nn.BCELoss()

    drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    standardSessions = np.array(['stage 5' in task and not any(variant in task for variant in ('nogo','noAR','oneReward','rewardOnly','catchOnly')) for task in df['task version']]) & ~np.array(df['ignore']).astype(bool)
    standardSessions = np.where(standardSessions)[0]
    sessionsToPass = getSessionsToPass(mouseId,df,standardSessions,stage=5)
    sessionStartTimes = df.loc[standardSessions,'start time'][sessionsToPass-2:sessionsToPass-2+21]
    sessionData = [getSessionData(mouseId,st) for st in sessionStartTimes]

    logLossTrain = []
    logLossTest = []
    prediction = []
    simulation = []
    simAction = []
    for testData in [sessionData[0]]:
        trainData = random.sample([s for s in sessionData if s is not testData],nTrainSessions)
        trainIndex = 0
        model = CustomRNN(inputSize,hiddenSize,outputSize,dropoutProb,hiddenType).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=learningRate,betas=smoothingConstants,weight_decay=weightDecay)
        logLossTrain.append(np.full(maxTrainIters,np.nan))
        logLossTest.append(np.full(maxTrainIters,np.nan))
        bestIter = 0
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
            logLossTrain[-1][i] = loss.item()
            
            model.eval()
            with torch.no_grad():
                session = testData
                modelInput,targetOutput = getModelInputAndTarget(session,inputSize,isFitToMouse,device)
                modelOutput = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
                logLossTest[-1][i] = lossFunc(modelOutput,targetOutput).item()
                if logLossTest[-1][i] < logLossTest[-1][bestIter]:
                    bestIter == i
                    bestModelStateDict = copy.deepcopy(model.state_dict())
                    
            if i > bestIter + earlyStopIters and np.all(logLossTest[-1][i-earlyStopIters:i+1] > logLossTest[-1][bestIter] + earlyStopThresh):
                break
        
        model.load_state_dict(bestModelStateDict)
        model.eval()
        with torch.no_grad():
            bestPred = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation)[0]
            bestSim = []
            bestSimAct = []
            for _ in range(10):
                pAction,action,reward = model(modelInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,isSimulation=True)
                bestSim.append(pAction)
                bestSimAct.append(action)
        prediction.append(bestPred.cpu().numpy())
        simulation.append([s.cpu().numpy() for s in bestSim])
        simAction.append([s.cpu().numpy() for s in bestSimAct])

    fileName = str(mouseId)+'_'+hiddenType+'_'+str(nTrainSessions)+'trainSessions_'+str(nHiddenUnits)+'hiddenUnits'+'.npz'
    filePath = os.path.join(baseDir,'Sam','RNNmodel',fileName)
    np.savez(filePath,sessions=[obj.startTime for obj in sessionData],
             logLossTrain=logLossTrain,logLossTest=logLossTest,prediction=prediction,simulation=simulation,simAction=simAction) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=str)
    parser.add_argument('--hiddenType',type=str)
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(1/25)
    torch.multiprocessing.set_start_method('spawn',force=True)
    processes = []
    for nTrainSessions in (4,8,12,16,20):
        for nHiddenUnits in ((4,8,16,32,64) if args.hiddenType=='rnn' else (2,4,8,16,32)):
            p = torch.multiprocessing.Process(target=trainModel,args=(args.mouseId,nTrainSessions,nHiddenUnits,args.hiddenType))
            processes.append(p)
            p.start()
    for p in processes:
        p.join() # wait for the process to complete
    