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
    if trainingPhase in ('initial training','after learning','clusters'):
        if trainingPhase == 'initial training':
            sessions = preExperimentSessions[:5]
        elif trainingPhase == 'after learning':
            sessionsToPass = getSessionsToPass(mouseId,df,preExperimentSessions,stage=5)
            sessions = preExperimentSessions[sessionsToPass:sessionsToPass+5]
        elif trainingPhase == 'clusters':
            sessions = preExperimentSessions
        testSession = sessions[sessionIndex]
        trainSessions = [s for s in sessions if s != testSession]
    else:
        sessions = np.array([trainingPhase in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
        sessions = np.where(sessions)[0]
        testSession = sessions[sessionIndex]
        trainSessions = preExperimentSessions[-4:]
    testData = getSessionData(mouseId,df.loc[testSession,'start time'])
    trainData = [getSessionData(mouseId,startTime) for startTime in df.loc[trainSessions,'start time']]
    if trainingPhase == 'clusters':
        clustData = np.load(os.path.join(baseDir,'Sam','clustData.npy'),allow_pickle=True).item()
        trainDataTrialCluster = [clustData['trialCluster'][str(mouseId)][startTime.strftime('%Y%m%d_%H%M%S')] for startTime in df.loc[trainSessions,'start time']]
    else:
        trainDataTrialCluster = None
    return testData,trainData,trainDataTrialCluster


def calcLogisticProb(q,beta,bias):
    return 1 / (1 + np.exp(-beta * (q - 0.5 + bias)))


def runModel(obj,betaAction,biasAction,biasAttention,visConfidence,audConfidence,
             wContext,alphaContext,decayContext,wReinforcement,alphaReinforcement,wHabit,alphaHabit,wReward,alphaReward,
             useHistory=True,nReps=1):

    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    modality = 0

    pContext = 0.5 + np.zeros((nReps,obj.nTrials,2))
    qContext = np.array([visConfidence,1-visConfidence,audConfidence,1-audConfidence])

    qReinforcement = np.zeros((nReps,obj.nTrials,len(stimNames)))
    qReinforcement[:,0] = qContext.copy()

    qHabit = qReinforcement.copy()

    qReward = np.zeros((nReps,obj.nTrials))

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

                vContext = np.sum(qContext * pStim * np.repeat(pContext[i,trial],2))

                qTotal[i,trial] = ((wContext * vContext) +
                                   (wReinforcement * np.sum(qReinforcement[i,trial] * pStim)) +
                                   (wHabit * np.sum(qHabit[i,trial] * pStim)) +
                                   (wReward * qReward[i,trial]))
                
                qTotal[i,trial] /= wContext + wReinforcement + wHabit + wReward

                pAction[i,trial] = calcLogisticProb(qTotal[i,trial],betaAction,biasAction)
                
                if useHistory:
                    action[i,trial] = obj.trialResponse[trial]
                elif random.random() < pAction[i,trial]:
                    action[i,trial] = 1 
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qReinforcement[i,trial+1] = qReinforcement[i,trial]
                qHabit[i,trial+1] = qHabit[i,trial]
                qReward[i,trial+1] = qReward[i,trial]

                outcome = obj.trialRewarded[trial]
                
                if action[i,trial]:
                    if alphaContext > 0:
                        pContext[i,trial+1,modality] += alphaContext * (outcome - vContext)
                        pContext[i,trial+1,modality] = np.clip(pContext[i,trial+1,modality],0,1)
                
                    if alphaReinforcement > 0:
                        qReinforcement[i,trial+1] += alphaReinforcement * pStim * (outcome - qReinforcement[i,trial])
                        qReinforcement[i,trial+1] = np.clip(qReinforcement[i,trial+1],0,1)

                if alphaHabit > 0:
                    qHabit[i,trial+1] += alphaHabit * pStim * (action[i,trial] - qHabit[i,trial])
                    qHabit[i,trial+1] += np.clip(qHabit[i,trial+1],0,1)

                if alphaReward > 0:
                    qReward[i,trial+1] += alphaReward * (outcome - qReward[i,trial])

                if decayContext > 0:
                    iti = (obj.trialStartTimes[trial+1] - obj.trialStartTimes[trial])
                    pContext[i,trial+1,modality] += (1 - np.exp(-iti/decayContext)) * (0.5 - pContext[i,trial+1,modality])
                pContext[i,trial+1,(1 if modality==0 else 0)] = 1 - pContext[i,trial+1,modality]
    
    return pContext, qReinforcement, qHabit, qReward, qTotal, pAction, action


def insertFixedParamVals(fitParams,fixedInd,fixedVal):
    nParams = len(fitParams) + (len(fixedVal) if isinstance(fixedVal,list) else 1)
    params = np.full(nParams,np.nan)
    params[fixedInd] = fixedVal
    params[np.isnan(params)] = fitParams
    return params


def evalModel(params,*args):
    trainData,trainDataTrialCluster,clust,fixedInd,fixedVal,modelTypeDict = args
    if fixedInd is not None:
        params = insertFixedParamVals(params,fixedInd,fixedVal)
    response = np.concatenate([obj.trialResponse for obj in trainData])
    prediction = np.concatenate([runModel(obj,*params,**modelTypeDict)[-2][0] for obj in trainData])
    if clust is not None:
        clustTrials = np.concatenate(trainDataTrialCluster) == clust
        response = response[clustTrials]
        prediction = prediction[clustTrials]
    logLoss = sklearn.metrics.log_loss(response,prediction)
    return logLoss


def fitModel(mouseId,trainingPhase,testData,trainData,trainDataTrialCluster):
    betaActionBounds = (0,40)
    biasActionBounds = (-1,1)
    biasAttentionBounds = (-1,1)
    visConfidenceBounds = (0.5,1)
    audConfidenceBounds = (0.5,1)
    wContextBounds = (0,1)
    alphaContextBounds = (0,1)
    decayContextBounds = (1,600) 
    wReinforcementBounds = (0,1)
    alphaReinforcementBounds = (0,1)
    wHabitBounds = (0,1)
    alphaHabitBounds = (0,1)
    wRewardBounds = (0,1)
    alphaRewardBounds = (0,1)

    bounds = (betaActionBounds,biasActionBounds,biasAttentionBounds,visConfidenceBounds,audConfidenceBounds,
              wContextBounds,alphaContextBounds,decayContextBounds,wReinforcementBounds,alphaReinforcementBounds,
              wHabitBounds,alphaHabitBounds,wRewardBounds,alphaRewardBounds)

    fixedValues = [None,0,0,1,1,0,0,0,0,0,0,0,0,0]

    modelTypeParamNames = ()
    modelTypeNames,modelTypes = zip(
                                    ('multiAgent', ()),
                                   )

    clustIds = np.arange(4)+1 if trainingPhase == 'clusters' else (None,)

    optParams = {'eps': 1e-4, 'maxfun': int(1e4),'maxiter': int(1e3),'locally_biased': True,'vol_tol': 1e-16,'len_tol': 1e-6}

    for modelTypeName,modelType in zip(modelTypeNames,modelTypes):
        fixedParamIndices = (None,1,2,3,4,[5,6,7],7,[8,9],[10,11],[12,13])
        fixedParamValues = [([fixedValues[j] for j in i] if isinstance(i,list) else (None if i is None else fixedValues[i])) for i in fixedParamIndices]
        modelTypeParams = {p: bool(m) for p,m in zip(modelTypeParamNames,modelType)}
        params = []
        logLoss = []
        terminationMessage = []
        for fixedInd,fixedVal in zip(fixedParamIndices,fixedParamValues):
            bnds = bounds if fixedInd is None else tuple(b for i,b in enumerate(bounds) if (i not in fixedInd if isinstance(fixedInd,list) else i != fixedInd))
            if trainingPhase == 'clusters':
                params.append([])
                logLoss.append([])
                terminationMessage.append([])
                prms = params[-1]
                nll = logLoss[-1]
                tm = terminationMessage[-1]
            else:
                prms = params
                nll = logLoss
                tm = terminationMessage
            for clust in clustIds:
                if clust is not None and not np.any(np.concatenate(trainDataTrialCluster) == clust):
                    prms.append(np.full((7 if modelTypeName == 'basicRL' else 10),np.nan))
                    nll.append(np.nan)
                    tm.append('')
                else:
                    fit = scipy.optimize.direct(evalModel,bnds,args=(trainData,trainDataTrialCluster,clust,fixedInd,fixedVal,modelTypeParams),**optParams)
                    prms.append((fit.x if fixedInd is None else insertFixedParamVals(fit.x,fixedInd,fixedVal)))
                    nll.append(fit.fun)
                    tm.append(fit.message)

        fileName = str(mouseId)+'_'+testData.startTime+'_'+trainingPhase+'_'+modelTypeName+'.npz'
        if trainingPhase == 'clusters':
            filePath = os.path.join(baseDir,'Sam','RLmodel','clusters',fileName)
        else:
            filePath = os.path.join(baseDir,'Sam','RLmodel',fileName)
        np.savez(filePath,params=params,logLoss=logLoss,terminationMessage=terminationMessage,**modelTypeParams) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=int)
    parser.add_argument('--sessionIndex',type=int)
    parser.add_argument('--trainingPhase',type=str)
    args = parser.parse_args()
    trainingPhase = args.trainingPhase.replace('_',' ')
    testData,trainData,trainDataTrialCluster = getSessionsToFit(args.mouseId,trainingPhase,args.sessionIndex)
    fitModel(args.mouseId,trainingPhase,testData,trainData,trainDataTrialCluster)
