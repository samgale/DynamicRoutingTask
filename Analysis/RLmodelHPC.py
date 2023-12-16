# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import argparse
import itertools
import os
import pathlib
import random
import numpy as np
import pandas as pd
import sklearn.metrics
from  DynamicRoutingAnalysisUtils import getFirstExperimentSession, getSessionsToPass, getSessionData


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting')


def getDataToFit(mouseId,trainingPhase,nSessions):
    drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
    firstExperimentSession = getFirstExperimentSession(df)
    if firstExperimentSession is not None:
        sessions[firstExperimentSession:] = False
    sessions = np.where(sessions)[0]
    if trainingPhase == 'initial training':
        sessions = sessions[:nSessions]
    else:
        sessionsToPass = getSessionsToPass(mouseId,df,sessions,stage=5)
        sessions = sessions[sessionsToPass:sessionsToPass+nSessions]
    sessionData = [getSessionData(mouseId,startTime) for startTime in df.loc[sessions,'start time']]
    return sessionData


def calcLogisticProb(q,tau,bias,norm=True):
    return 1 / (1 + np.exp(-(q + bias) / tau))


def runModel(obj,contextMode,visConfidence,audConfidence,alphaContext,tauAction,biasAction,alphaAction,penalty,nIters=20):
    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    
    pContext = np.zeros((nIters,obj.nTrials,2),dtype=float) + 0.5
    
    qAction = np.zeros((nIters,obj.nTrials,2,len(stimNames)),dtype=float) + penalty
    if contextMode == 'no context':
        qAction[:,:,:,[0,2]] = 1
    else:
        qAction[:,:,0,0] = 1
        qAction[:,:,1,2] = 1

    expectedValue = np.zeros((nIters,obj.nTrials))
    
    response = np.zeros((nIters,obj.nTrials),dtype=int)
    
    for i in range(nIters):
        for trial,(stim,rewStim,autoRew) in enumerate(zip(obj.trialStim,obj.rewardedStim,obj.autoRewardScheduled)):
            if stim == 'catch':
                action = 0
            else:
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]
                
                if contextMode == 'switch context':
                    if trial == 0:
                        context = modality
                    else:
                        context = 0 if pContext[i,trial,0] > 0.5 else 1
                else:
                    context = 0
                    
                if contextMode == 'weight context':
                    expectedValue[i,trial] = np.sum(qAction[i,trial] * pStim[None,:] * pContext[i,trial][:,None])
                else:
                    expectedValue[i,trial] = np.sum(qAction[i,trial,context] * pStim)
                pAction = calcLogisticProb(expectedValue[i,trial],tauAction,biasAction)
                
                action = 1 if autoRew or random.random() < pAction else 0 
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qAction[i,trial+1] = qAction[i,trial]
            
                if action:
                    outcome = 1 if stim==rewStim else penalty
                    predictionError = outcome - expectedValue[i,trial]
                    
                    if contextMode != 'no context':
                        if outcome < 1:
                            pContext[i,trial+1,modality] -= alphaContext * pStim[0 if modality==0 else 2] * pContext[i,trial,modality]
                        else:
                            pContext[i,trial+1,modality] += alphaContext * (1 - pContext[i,trial,modality]) 
                        pContext[i,trial+1,1 if modality==0 else 0] = 1 - pContext[i,trial+1,modality]
                    
                    if contextMode == 'weight context':
                        qAction[i,trial+1] += alphaAction * pStim[None,:] * pContext[i,trial][:,None] * predictionError
                    else:
                        qAction[i,trial+1,context] += alphaAction * pStim * predictionError
                    qAction[i,trial+1][qAction[i,trial+1] > 1] = 1 
                    qAction[i,trial+1][qAction[i,trial+1] < penalty] = penalty 
            
            response[i,trial] = action
    
    return response, pContext, qAction, expectedValue


def fitModel(mouseId,sessionData,sessionIndex,trainingPhase,contextMode,qMode,nJobs,jobIndex):
    visConfidenceRange = np.arange(0.6,1.01,0.05)
    audConfidenceRange = np.arange(0.6,1.01,0.05)
    if contextMode == 'no context':
        alphaContextRange = (0,)
    else:
        alphaContextRange = np.arange(0.05,1,0.1) 
    tauActionRange = np.arange(0.05,0.5,0.05)
    biasActionRange = np.arange(0,1,0.1)
    if qMode == 'no q update':
        alphaActionRange = (0,)
    else:
        alphaActionRange = np.arange(0.01,0.15,0.01) if trainingPhase=='initial training' else np.arange(0.05,1,0.1)
    penaltyRange = np.arange(-1,0,0.2)

    fitParamRanges = (visConfidenceRange,audConfidenceRange,alphaContextRange,
                      tauActionRange,biasActionRange,alphaActionRange,penaltyRange)
    fitParamsIter = itertools.product(*fitParamRanges)
    nParamCombos = np.prod([len(p) for p in fitParamRanges])
    paramCombosPerJob = int(np.ceil(nParamCombos/nJobs))
    paramsStart = jobIndex * paramCombosPerJob
    
    testExp = sessionData[sessionIndex]
    trainExps = [obj for obj in sessionData if obj is not testExp]
    actualResponse = np.concatenate([obj.trialResponse for obj in trainExps])
    minLogLoss = None
    for params in itertools.islice(fitParamsIter,paramsStart,paramsStart+paramCombosPerJob):
        modelResponse = np.concatenate([np.mean(runModel(obj,contextMode,*params)[0],axis=0) for obj in trainExps])
        logLoss = sklearn.metrics.log_loss(actualResponse,modelResponse)
        if minLogLoss is None or logLoss < minLogLoss:
            minLogLoss = logLoss
            bestParams = params

    fileName = str(mouseId)+'_'+testExp.startTime+'_'+trainingPhase+'_'+contextMode+'_'+qMode+'_job'+str(jobIndex)+'.npz'
    filePath = os.path.join(baseDir,'Sam','RLmodel',fileName)
    np.savez(filePath,params=bestParams,logLoss=minLogLoss)  
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=int)
    parser.add_argument('--nSessions',type=int)
    parser.add_argument('--sessionIndex',type=int)
    parser.add_argument('--trainingPhase',type=str)
    parser.add_argument('--contextMode',type=str)
    parser.add_argument('--qMode',type=str)
    parser.add_argument('--nJobs',type=int)
    parser.add_argument('--jobIndex',type=int)
    args = parser.parse_args()
    trainingPhase,contextMode,qMode = [a.replace('_',' ') for a in (args.trainingPhase,args.contextMode,args.qMode)]
    sessionData = getDataToFit(args.mouseId,trainingPhase,args.nSessions)
    fitModel(args.mouseId,sessionData,args.sessionIndex,trainingPhase,contextMode,qMode,args.nJobs,args.jobIndex)
