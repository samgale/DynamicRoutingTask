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


def getSessionsToFit(mouseId,trainingPhase,sessionIndex):
    drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    preExperimentSessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
    firstExperimentSession = getFirstExperimentSession(df)
    if firstExperimentSession is not None:
        preExperimentSessions[firstExperimentSession:] = False
    preExperimentSessions = np.where(preExperimentSessions)[0]
    if trainingPhase in ('initial training','after learning'):
        if trainingPhase == 'initial training':
            sessions = preExperimentSessions[:5]
        elif trainingPhase == 'after learning':
            sessionsToPass = getSessionsToPass(mouseId,df,preExperimentSessions,stage=5)
            sessions = preExperimentSessions[sessionsToPass:sessionsToPass+5]
        testSession = sessions[sessionIndex]
        trainSessions = [s for s in sessions if s != testSession]
    else:
        sessions = np.array([trainingPhase in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
        sessions = np.where(sessions)[0]
        testSession = sessions[sessionIndex]
        trainSessions = preExperimentSessions[-4:]
    testData = getSessionData(mouseId,df.loc[testSession,'start time'])
    trainData = [getSessionData(mouseId,startTime) for startTime in df.loc[trainSessions,'start time']]
    return testData,trainData


def calcLogisticProb(q,tau,bias):
    return 1 / (1 + np.exp(-(q + bias) / tau))


def runModel(obj,tauAction,biasAction,visConfidence,audConfidence,alphaContext,alphaAction,alphaHabit,
             weightContext=False,weightAction=False,weightLearning=False,attendReward=False,useRPE=False,
             useHistory=True,nReps=1):
    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]

    pContext = 0.5 + np.zeros((nReps,obj.nTrials,2))
    qContext = -np.ones((nReps,obj.nTrials,2,len(stimNames)))  
    qContext[:,:,0,:2] = 2 * np.array([visConfidence,1-visConfidence]) - 1
    qContext[:,:,1,-2:] = 2 * np.array([audConfidence,1-audConfidence]) - 1

    qStim = -np.ones((nReps,obj.nTrials,len(stimNames)))

    wHabit = np.zeros((nReps,obj.nTrials))
    if alphaHabit > 0:
        wHabit += 0.5
    qHabit = np.array([2 * visConfidence - 1,
                       2 * (1-visConfidence) - 1,
                       2 * audConfidence - 1,
                       2 * (1-audConfidence) - 1])

    expectedValue = -np.ones((nReps,obj.nTrials))

    pAction = np.zeros((nReps,obj.nTrials))
    
    action = np.zeros((nReps,obj.nTrials),dtype=int)
    
    for i in range(nReps):
        for trial,stim in enumerate(obj.trialStim):
            if stim != 'catch':
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]

                if weightAction:
                    expectedValue[i,trial] = np.sum(qStim[i,trial] * pStim * np.repeat(pContext[i,trial],2))
                else:
                    valContext = np.sum(qContext[i,trial] * pStim[None,:] * pContext[i,trial][:,None])
                    valStim = np.sum(qStim[i,trial] * pStim)
                    if attendReward:
                        expectedValue[i,trial] = valStim
                    if weightContext:
                        wContext = 1 - 2 * pContext[i,trial].min()
                        expectedValue[i,trial] = wContext * valContext  + (1 - wContext) * valStim 
                    else:   
                        if alphaContext > 0:
                            expectedValue[i,trial] = valContext
                        else:
                            expectedValue[i,trial] = valStim

                q = (wHabit[i,trial] * np.sum(qHabit * pStim)) + ((1 - wHabit[i,trial]) * expectedValue[i,trial])           
            
                pAction[i,trial] = calcLogisticProb(q,tauAction,biasAction)
                
                if useHistory:
                    action[i,trial] = obj.trialResponse[trial]
                elif random.random() < pAction[i,trial]:
                    action[i,trial] = 1 
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qContext[i,trial+1] = qContext[i,trial]
                qStim[i,trial+1] = qStim[i,trial]
                wHabit[i,trial+1] = wHabit[i,trial]
            
                if action[i,trial] or obj.autoRewarded[trial]:
                    outcome = 1 if obj.trialRewarded[trial] else -1
                    predictionError = outcome - expectedValue[i,trial]
                    
                    if alphaContext > 0 and stim != 'catch':
                        if attendReward:
                            contextError = outcome
                        elif useRPE:
                            contextError = 0.5 * predictionError
                        else:
                            if outcome < 1:
                                contextError = -1 * pStim[0 if modality==0 else 2] * pContext[i,trial,modality]
                            else:
                                contextError = 1 - pContext[i,trial,modality] 
                        pContext[i,trial+1,modality] += alphaContext * contextError
                        pContext[i,trial+1,modality] = np.clip(pContext[i,trial+1,modality],0,1)
                        pContext[i,trial+1,(1 if modality==0 else 0)] = 1 - pContext[i,trial+1,modality]
                    
                    if alphaAction > 0 and stim != 'catch':
                        if attendReward or weightAction or weightContext or alphaContext == 0:
                            dq = alphaAction * pStim * predictionError
                            if weightLearning:
                                dq *= pContext[i,trial,modality]
                            qStim[i,trial+1] += dq
                            qStim[i,trial+1] = np.clip(qStim[i,trial+1],-1,1)
                        else:
                            qContext[i,trial+1] += alphaAction * pContext[i,trial][:,None] * pStim[None,:] * predictionError
                            qContext[i,trial+1] = np.clip(qContext[i,trial+1],-1,1)

                    if alphaHabit > 0:
                        wHabit[i,trial+1] += alphaHabit * (0.5 * abs(predictionError) - wHabit[i,trial])
    
    return pContext, qContext, qStim, wHabit, expectedValue, pAction, action


def evalModel(params,*args):
    trainData,fixedValInd,fixedVal,modelTypeDict = args
    if fixedVal is not None:
        params = np.insert(params,(fixedValInd[0] if isinstance(fixedValInd,tuple) else fixedValInd),fixedVal)
    response = np.concatenate([obj.trialResponse for obj in trainData])
    prediction = np.concatenate([runModel(obj,*params,**modelTypeDict)[-2][0] for obj in trainData])
    logLoss = sklearn.metrics.log_loss(response,prediction)
    return logLoss


def fitModel(mouseId,trainingPhase,testData,trainData):
    tauActionBounds = (0.01,1)
    biasActionBounds = (-1,1)
    visConfidenceBounds = (0.5,1)
    audConfidenceBounds = (0.5,1)
    alphaContextBounds = (0,1) 
    alphaActionBounds = (0,1)
    alphaHabitBounds = (0,1)

    bounds = (tauActionBounds,biasActionBounds,visConfidenceBounds,audConfidenceBounds,
              alphaContextBounds,alphaActionBounds,alphaHabitBounds)

    fixedValueIndices = (None,1,2,3,4,5,(4,5),6)
    fixedValues = (None,0,1,1,0,0,(0,0),0)

    modelTypeParamNames = ('weightContext','weightAction','weightLearning','attendReward','useRPE')
    modelTypeNames,modelTypes = zip(
                                    ('contextQ',(0,0,0,0,1)),
                                    ('weightContext',(1,0,0,0,1)),
                                    ('weightAction',(0,1,0,0,1)),
                                    ('attendActionLearn',(0,1,1,1,0)),
                                    ('attendAction',(0,1,0,1,0)),
                                    ('attendLearn',(0,0,1,1,0)),
                                   )

    for modelTypeName,modelType in zip(modelTypeNames,modelTypes):
        modelTypeParams = {p: bool(m) for p,m in zip(modelTypeParamNames,modelType)}
        fit = scipy.optimize.direct(evalModel,bounds,args=(trainData,None,None,modelTypeParams))
        params = [fit.x]
        logLoss = [fit.fun]
        for fixedValInd,fixedVal in zip(fixedValueIndices,fixedValues):
            if fixedVal is not None:
                bnds = tuple(b for i,b in enumerate(bounds) if (i not in fixedValInd if isinstance(fixedValInd,tuple) else i != fixedValInd))
                fit = scipy.optimize.direct(evalModel,bnds,args=(trainData,fixedValInd,fixedVal,modelTypeParams))
                params.append(np.insert(fit.x,(fixedValInd[0] if isinstance(fixedValInd,tuple) else fixedValInd),fixedVal))
                logLoss.append(fit.fun)

        fileName = str(mouseId)+'_'+testData.startTime+'_'+trainingPhase+'_'+modelTypeName+'.npz'
        filePath = os.path.join(baseDir,'Sam','RLmodel',fileName)
        np.savez(filePath,params=params,logLoss=logLoss,**modelTypeParams) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=int)
    parser.add_argument('--sessionIndex',type=int)
    parser.add_argument('--trainingPhase',type=str)
    args = parser.parse_args()
    trainingPhase = args.trainingPhase.replace('_',' ')
    testData,trainData = getSessionsToFit(args.mouseId,trainingPhase,args.sessionIndex)
    fitModel(args.mouseId,trainingPhase,testData,trainData)
