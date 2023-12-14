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
import h5py
import numpy as np
import pandas as pd
import sklearn.metrics
from  DynamicRoutingAnalysisUtils import getFirstExperimentSession, getSessionsToPass, getSessionData


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting')
drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)


def dictToHdf5(group,d):
    for key,val in d.items():
        if isinstance(val,dict): 
            dictToHdf5(group.create_group(key),val)
        else:
            if (hasattr(val,'__len__') and len(val) > 0 and all([hasattr(v,'__len__') for v in val])
                and [len(v) for v in val].count(len(val[0])) != len(val)):
                group.create_dataset(key,data=np.array(val,dtype=object),dtype=h5py.special_dtype(vlen=float))
            else:
                group.create_dataset(key,data=val)


def calcLogisticProb(q,tau,bias,norm=True):
    return 1 / (1 + np.exp(-(q + bias) / tau))


def runModel(obj,contextMode,visConfidence,audConfidence,alphaContext,tauAction,biasAction,alphaAction,penalty,nIters=10):
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


def fitModel(mouseId,nSessions,sessionIndex,trainingPhase,contextMode,qMode,nJobs,jobIndex):
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = np.array(['stage 5' in task for task in df['task version']])
    firstExperimentSession = getFirstExperimentSession(df)
    if firstExperimentSession is not None:
        sessions[firstExperimentSession:] = False
    sessions = np.where(sessions)[0]
    if trainingPhase == 'initial training':
        sessions = sessions[:nSessions]
    else:
        sessionsToPass = getSessionsToPass(mouseId,df,stage=5)
        sessions = sessions[sessionsToPass:sessionsToPass+nSessions]
    exps = getSessionData(mouseId,df.iloc[sessions])
        
    visConfidenceRange = np.arange(0.6,1.01,0.1)
    audConfidenceRange = np.arange(0.6,1.01,0.1)
    if contextMode == 'no context':
        alphaContextRange = (0,)
    else:
        alphaContextRange = np.arange(0.05,1,0.15) 
    tauActionRange = (0.1,0.5,0.1)
    biasActionRange = np.arange(0,1,0.15)
    if qMode == 'no q update':
        alphaActionRange = (0,)
    else:
        alphaActionRange = np.arange(0.02,0.13,0.02) if trainingPhase=='initial training' else np.arange(0.05,1,0.15)
    penaltyRange = (-1,) #np.arange(-1,0,0.2)

    fitParamRanges = (visConfidenceRange,audConfidenceRange,alphaContextRange,
                      tauActionRange,biasActionRange,alphaActionRange,penaltyRange)
    fitParamsIter = itertools.product(*fitParamRanges)
    nParamCombos = np.prod([len(p) for p in fitParamRanges])
    paramCombosPerJob = int(np.ceil(nParamCombos/nJobs))
    paramsStart = jobIndex * paramCombosPerJob
    
    testExp = exps[sessionIndex]
    trainExps = [obj for obj in exps if obj is not testExp]
    actualResponse = np.concatenate([obj.trialResponse for obj in trainExps])
    minLogLoss = 1000
    for params in itertools.islice(fitParamsIter,paramsStart,paramsStart+paramCombosPerJob):
        modelResponse = np.concatenate([np.mean(runModel(obj,contextMode,*params)[0],axis=0) for obj in trainExps])
        logLoss = sklearn.metrics.log_loss(actualResponse,modelResponse)
        if logLoss < minLogLoss:
            minLogLoss = logLoss
            bestParams = params

    fileName = str(mouseId)+'_'+'session'+str(sessionIndex)+'_'+trainingPhase+'_'+contextMode+'_'+qMode+'job'+str(jobIndex)+'.npz'
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
    fitModel(args.mouseId,args.nSessions,args.sessionIndex,
             args.trainingPhase.replace('_',' '),args.contextMode.replace('_',' '),args.qMode.replace('_',' '),
             args.nJobs,args.jobIndex)
