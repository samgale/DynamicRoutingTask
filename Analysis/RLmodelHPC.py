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


def calcLogisticProb(q,beta,bias):
    return 1 / (1 + np.exp(-beta * (q - 0.5 + bias)))


def runModel(obj,betaAction,biasAction,biasAttention,visConfidence,audConfidence,alphaContext,alphaAction,
             decayContext,alphaHabit,alphaReward,weightAttention=False,useRPE=False,useHistory=True,nReps=1):

    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    modality = 0

    pContext = 0.5 + np.zeros((nReps,obj.nTrials,2))

    qContext = np.zeros((nReps,obj.nTrials,2,len(stimNames))) 
    if alphaContext > 0:
        qContext[:,:,0,:2] = [visConfidence,1-visConfidence]
        qContext[:,:,1,-2:] = [audConfidence,1-audConfidence]

    qStim = np.zeros((nReps,obj.nTrials,len(stimNames)))
    if alphaAction > 0:
        qStim[:,:] = [visConfidence,1-visConfidence,audConfidence,1-audConfidence]

    wHabit = np.zeros((nReps,obj.nTrials))
    if alphaHabit > 0:
        wHabit += 0.5
    qHabit = np.array([visConfidence,1-visConfidence,audConfidence,1-audConfidence])

    wReward = np.zeros((nReps,obj.nTrials))

    expectedValue = np.zeros((nReps,obj.nTrials))

    qTotal = np.zeros((nReps,obj.nTrials))

    pAction = np.zeros((nReps,obj.nTrials))
    
    action = np.zeros((nReps,obj.nTrials),dtype=int)
    
    for i in range(nReps):
        for trial,stim in enumerate(obj.trialStim):
            if stim != 'catch':
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]
                if biasAttention > 0:
                    pStim[-2:] *= 1 - biasAttention
                else:
                    pStim[:2] *= 1 + biasAttention

                if weightAttention:
                    expectedValue[i,trial] = np.sum(qStim[i,trial] * pStim * np.repeat(pContext[i,trial],2))
                elif alphaContext > 0:
                    expectedValue[i,trial] = np.sum(qContext[i,trial] * pStim[None,:] * pContext[i,trial][:,None])
                else:
                    expectedValue[i,trial] = np.sum(qStim[i,trial] * pStim)

                qTotal[i,trial] = wReward[i,trial] + (wHabit[i,trial] * np.sum(qHabit * pStim)) + ((1 - wHabit[i,trial]) * expectedValue[i,trial])           
                
                pAction[i,trial] = calcLogisticProb(qTotal[i,trial],betaAction,biasAction)
                
                if useHistory:
                    action[i,trial] = obj.trialResponse[trial]
                elif random.random() < pAction[i,trial]:
                    action[i,trial] = 1 
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qContext[i,trial+1] = qContext[i,trial]
                qStim[i,trial+1] = qStim[i,trial]
                wHabit[i,trial+1] = wHabit[i,trial]
                wReward[i,trial+1] = wReward[i,trial]
            
                if action[i,trial] or obj.autoRewarded[trial]:
                    outcome = 1 if stim == obj.rewardedStim[trial] or obj.autoRewarded[trial] else 0
                    predictionError = outcome - expectedValue[i,trial]
                    
                    if alphaContext > 0:
                        if stim != 'catch':
                            if useRPE:
                                contextError = predictionError
                            else:
                                if outcome:
                                    contextError = 1 - pContext[i,trial,modality]
                                else:
                                    contextError = -pContext[i,trial,modality] * pStim[0 if modality==0 else 2]
                            pContext[i,trial+1,modality] += alphaContext * contextError
                            pContext[i,trial+1,modality] = np.clip(pContext[i,trial+1,modality],0,1)
                    
                    if alphaAction > 0 and stim != 'catch':
                        if weightAttention or alphaContext == 0:
                            qStim[i,trial+1] += alphaAction * pStim * predictionError
                            qStim[i,trial+1] = np.clip(qStim[i,trial+1],0,1)
                        else:
                            qContext[i,trial+1] += alphaAction * pContext[i,trial][:,None] * pStim[None,:] * predictionError
                            qContext[i,trial+1] = np.clip(qContext[i,trial+1],0,1)

                    if alphaHabit > 0:
                        wHabit[i,trial+1] += alphaHabit * (abs(predictionError) - wHabit[i,trial])

                    if alphaReward > 0 and outcome > 0:
                        wReward[i,trial+1] += alphaReward * (outcome - wReward[i,trial])

                if decayContext > 0:
                    iti = (obj.trialStartTimes[trial+1] - obj.trialStartTimes[trial])
                    pContext[i,trial+1,modality] += (1 - np.exp(-iti/decayContext)) * (0.5 - pContext[i,trial+1,modality])
                    pContext[i,trial+1,(1 if modality==0 else 0)] = 1 - pContext[i,trial+1,modality]

                if alphaReward > 0:
                    wReward[i,trial+1] -= alphaReward * wReward[i,trial]
    
    return pContext, qContext, qStim, wHabit, wReward, expectedValue, qTotal, pAction, action


def insertFixedParamVals(fitParams,fixedValInd,fixedVal):
    nParams = len(fitParams) + (len(fixedVal) if isinstance(fixedVal,list) else 1)
    params = np.full(nParams,np.nan)
    params[fixedValInd] = fixedVal
    params[np.isnan(params)] = fitParams
    return params


def evalModel(params,*args):
    trainData,fixedValInd,fixedVal,modelTypeDict = args
    if fixedVal is not None:
        params = insertFixedParamVals(params,fixedValInd,fixedVal)
    response = np.concatenate([obj.trialResponse for obj in trainData])
    prediction = np.concatenate([runModel(obj,*params,**modelTypeDict)[-2][0] for obj in trainData])
    logLoss = sklearn.metrics.log_loss(response,prediction)
    return logLoss


def fitModel(mouseId,trainingPhase,testData,trainData):
    betaActionBounds = (0,40)
    biasActionBounds = (-1,1)
    biasAttentionBounds  = (-1,1)
    visConfidenceBounds = (0.5,1)
    audConfidenceBounds = (0.5,1)
    alphaContextBounds = (0,1)
    alphaActionBounds = (0,1)
    decayContextBounds = (1,600) 
    alphaHabitBounds = (0,1)
    alphaRewardBounds = (0,1)

    bounds = (betaActionBounds,biasActionBounds,biasAttentionBounds,visConfidenceBounds,audConfidenceBounds,
              alphaContextBounds,alphaActionBounds,decayContextBounds,alphaHabitBounds,alphaRewardBounds)

    fixedValueIndices = (None,1,2,3,4,[5,7],6,[5,6,7],[5,7,8],7,8,[7,8],9)
    fixedValues = (None,0,0,1,1,[0,0],0,[0,0,0],[0,0,0],0,0,[0,0],0)

    modelTypeParamNames = ('weightAttention','useRPE')
    modelTypeNames,modelTypes = zip(
                                    ('contextQ',(0,0)),
                                    #('contextQRPE',(0,1)),
                                    #('weightAttention',(1,0)),
                                    #('weightAttentionRPE',(0,1)),
                                   )

    optParams = {'eps': 1e-4, 'maxfun': int(1e4),'maxiter': int(1e3),'locally_biased': True,'vol_tol': 1e-16,'len_tol': 1e-6}

    for modelTypeName,modelType in zip(modelTypeNames,modelTypes):
        modelTypeParams = {p: bool(m) for p,m in zip(modelTypeParamNames,modelType)}
        fit = scipy.optimize.direct(evalModel,bounds,args=(trainData,None,None,modelTypeParams),**optParams)
        params = [fit.x]
        logLoss = [fit.fun]
        terminationMessage = [fit.message]
        for fixedValInd,fixedVal in zip(fixedValueIndices,fixedValues):
            if modelTypeName == 'proHabit' and (8 not in fixedValInd if isinstance(fixedValInd,list) else fixedValInd != 8):
                if fixedValInd is None:
                    fixedValInd,fixedVal = (8,0)
                elif isinstance(fixedValInd,list):
                    fixedValInd,fixedVal = (fixedValInd+[8],fixedVal+[0])
                else:
                    fixedValInd,fixedVal = ([fixedValInd,8],[fixedVal,0])
            if fixedVal is not None:
                bnds = tuple(b for i,b in enumerate(bounds) if (i not in fixedValInd if isinstance(fixedValInd,list) else i != fixedValInd))
                fit = scipy.optimize.direct(evalModel,bnds,args=(trainData,fixedValInd,fixedVal,modelTypeParams),**optParams)
                params.append(insertFixedParamVals(fit.x,fixedValInd,fixedVal))
                logLoss.append(fit.fun)
                terminationMessage.append(fit.message)

        fileName = str(mouseId)+'_'+testData.startTime+'_'+trainingPhase+'_'+modelTypeName+'.npz'
        filePath = os.path.join(baseDir,'Sam','RLmodel',fileName)
        np.savez(filePath,params=params,logLoss=logLoss,terminationMessage=terminationMessage,**modelTypeParams) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=int)
    parser.add_argument('--sessionIndex',type=int)
    parser.add_argument('--trainingPhase',type=str)
    args = parser.parse_args()
    trainingPhase = args.trainingPhase.replace('_',' ')
    testData,trainData = getSessionsToFit(args.mouseId,trainingPhase,args.sessionIndex)
    fitModel(args.mouseId,trainingPhase,testData,trainData)
