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
import scipy.optimize
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


def calcLogisticProb(q,tau,bias):
    return 1 / (1 + np.exp(-(q + bias) / tau))


def calcNormLogisticProb(q,tau,bias):
    p = calcLogisticProb(q,tau,bias)
    low = calcLogisticProb(-1,tau,bias)
    high = calcLogisticProb(1,tau,bias)
    p -= low
    p /= high-low
    return p


def runModel(obj,tauAction,biasAction,visConfidence,audConfidence,alphaContext,alphaAction,useHistory=True,nReps=1):
    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    
    pContext = np.zeros((nReps,obj.nTrials,2)) + 0.5
    
    qAction = -np.ones((nReps,obj.nTrials,2,len(stimNames)),dtype=float)  
    if alphaContext > 0:
        qAction[:,:,0,0] = 1
        qAction[:,:,1,2] = 1
    else:
        qAction[:,:,:,[0,2]] = 1

    expectedValue = np.zeros((nReps,obj.nTrials))

    pAction = np.zeros((nReps,obj.nTrials))
    
    action = np.zeros((nReps,obj.nTrials),dtype=int)
    
    for i in range(nReps):
        for trial,(stim,rewStim,autoRew) in enumerate(zip(obj.trialStim,obj.rewardedStim,obj.autoRewardScheduled)):
            if stim != 'catch':
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]
                    
                if alphaContext > 0:
                    expectedValue[i,trial] = np.sum(qAction[i,trial] * pStim[None,:] * pContext[i,trial][:,None])
                else:
                    context = 0
                    expectedValue[i,trial] = np.sum(qAction[i,trial,context] * pStim)
                pAction[i,trial] = calcLogisticProb(expectedValue[i,trial],tauAction,biasAction)
                
                if autoRew:
                    action[i,trial] = 1
                elif useHistory:
                    action[i,trial] = obj.trialResponse[trial]
                else:
                    action[i,trial] = 1 if random.random() < pAction[i,trial] else 0 
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qAction[i,trial+1] = qAction[i,trial]
            
                if action[i,trial]:
                    outcome = 1 if stim==rewStim else -1
                    predictionError = outcome - expectedValue[i,trial]
                    
                    if alphaContext > 0:
                        if outcome < 1:
                            pContext[i,trial+1,modality] -= alphaContext * pStim[0 if modality==0 else 2] * pContext[i,trial,modality]
                        else:
                            pContext[i,trial+1,modality] += alphaContext * (1 - pContext[i,trial,modality]) 
                        pContext[i,trial+1,1 if modality==0 else 0] = 1 - pContext[i,trial+1,modality]
                    
                    if alphaAction > 0:
                        if alphaContext > 0:
                            qAction[i,trial+1] += alphaAction * pStim[None,:] * pContext[i,trial][:,None] * predictionError
                        else:
                            qAction[i,trial+1,context] += alphaAction * pStim * predictionError
                        qAction[i,trial+1][qAction[i,trial+1] > 1] = 1 
                        qAction[i,trial+1][qAction[i,trial+1] < -1] = -1 
    
    return pContext, qAction, expectedValue, pAction, action


def evalModel(params,*args):
    trainExps,fixedValInd,fixedVal = args
    if fixedVal is not None:
        params = np.insert(params,fixedValInd,fixedVal)
    actualResponse = np.concatenate([obj.trialResponse for obj in trainExps])
    pAction = np.concatenate([runModel(obj,*params)[3][0] for obj in trainExps])
    print(params)
    logLoss = sklearn.metrics.log_loss(actualResponse,pAction)
    return logLoss


def fitModel(mouseId,sessionData,sessionIndex,trainingPhase):
    testExp = sessionData[sessionIndex]
    trainExps = [obj for obj in sessionData if obj is not testExp]

    tauActionBounds = (0.01,1)
    biasActionBounds = (-1,1)
    visConfidenceBounds = (0.5,1)
    audConfidenceBounds = (0.5,1)
    alphaContextBounds = (0,1) 
    alphaActionBounds = (0,1)

    bounds = (tauActionBounds,biasActionBounds,visConfidenceBounds,audConfidenceBounds,alphaContextBounds,alphaActionBounds)

    fixedValues = (None,None,1,1,0,0)

    fit = scipy.optimize.direct(evalModel,bounds,args=(trainExps,None,None))
    params = [fit.x]
    logLoss = [fit.fun]
    for fixedValInd,fixedVal in enumerate(fixedValues):
        if fixedVal is not None:
            bnds = tuple(b for i,b in enumerate(bounds) if i != fixedValInd)
            fit = scipy.optimize.direct(evalModel,bnds,args=(trainExps,fixedValInd,fixedVal))
            params.append(np.insert(fit.x,fixedValInd,fixedVal))
            logLoss.append(fit.fun)

    fileName = str(mouseId)+'_'+testExp.startTime+'_'+trainingPhase+'.npz'
    filePath = os.path.join(baseDir,'Sam','RLmodel',fileName)
    np.savez(filePath,params=params,logLoss=logLoss) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=int)
    parser.add_argument('--nSessions',type=int)
    parser.add_argument('--sessionIndex',type=int)
    parser.add_argument('--trainingPhase',type=str)
    args = parser.parse_args()
    trainingPhase = args.trainingPhase.replace('_',' ')
    sessionData = getDataToFit(args.mouseId,trainingPhase,args.nSessions)
    fitModel(args.mouseId,sessionData,args.sessionIndex,trainingPhase)
