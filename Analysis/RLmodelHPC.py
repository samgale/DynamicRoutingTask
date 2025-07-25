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
import scipy.stats
import sklearn.metrics
import psytrack
import ssm
from  DynamicRoutingAnalysisUtils import getFirstExperimentSession, getSessionsToPass, getSessionData


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting')


def getSessionsToFit(mouseId,trainingPhase,sessionIndex):
    if trainingPhase == 'opto':
        optoLabel = 'lFC'
        df = pd.read_excel(os.path.join(baseDir,'Sam','OptoExperiments.xlsx'),sheet_name=str(mouseId))
        sessions = np.where(df[optoLabel] & ~(df['unilateral'] & df['bilateral']))[0]
    else:
        drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
        df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
        ignore = np.array(df['ignore']).astype(bool)
        if trainingPhase in ('initial training','after learning','clusters'):
            preExperimentSessions = np.array(['stage 5' in task for task in df['task version']]) & ~ignore
            firstExperimentSession = getFirstExperimentSession(df)
            if firstExperimentSession is not None:
                preExperimentSessions[firstExperimentSession:] = False
            preExperimentSessions = np.where(preExperimentSessions)[0]
            if trainingPhase == 'initial training':
                sessions = preExperimentSessions[:5]
            elif trainingPhase == 'after learning':
                sessionsToPass = getSessionsToPass(mouseId,df,preExperimentSessions,stage=5)
                sessions = preExperimentSessions[sessionsToPass:sessionsToPass+5]
            elif trainingPhase == 'clusters':
                sessions = preExperimentSessions
        elif trainingPhase == 'ephys':
            ephys = np.where(df['ephys'] & ~ignore)[0]
            hab = np.where(df['hab'] & ~ignore)[0][-2:]
            sessions = np.concatenate((ephys,hab))
        else:
            sessions = np.array([trainingPhase in task for task in df['task version']]) & ~ignore
            sessions = np.where(sessions)[0]
    testSession = sessions[sessionIndex]
    testData = getSessionData(mouseId,df.loc[testSession,'start time'])
    trainSessions = [s for s in sessions if s != testSession]
    trainData = [getSessionData(mouseId,startTime) for startTime in df.loc[trainSessions,'start time']]
    return testData,trainData


# def runModel(obj,visConfidence,audConfidence,biasAction,
#              wContext,alphaContext,alphaContextNeg,tauContext,blockTiming,blockTimingShape,
#              wReinforcement,alphaReinforcement,alphaReinforcementNeg,tauReinforcement,
#              wPerseveration,alphaPerseveration,tauPerseveration,alphaReward,tauReward,
#              stateSpace=False,contextPerseveration=False,initReinforcement=False,initPerseveration=False,
#              scalarError=False,optoLabel=None,useChoiceHistory=True,nReps=1):

#     stimNames = ('vis1','vis2','sound1','sound2')
#     stimConfidence = [visConfidence,audConfidence]
#     modality = 0

#     pContext = 0.5 + np.zeros((nReps,obj.nTrials,2))
#     qContext = np.array([1,0,1,0])

#     if stateSpace:
#         qReinforcement = np.zeros((nReps,obj.nTrials,2,len(stimNames)))
#         if initReinforcement:
#             qReinforcement[:,0] = [[1,0,0,0],[0,0,1,0]]
#     else:
#         qReinforcement = np.zeros((nReps,obj.nTrials,len(stimNames)))
#         if initReinforcement:
#             qReinforcement[:,0] = [1,0,1,0]

#     if stateSpace and contextPerseveration:
#         qPerseveration = np.zeros((nReps,obj.nTrials,2,len(stimNames)))
#         if initPerseveration:
#             qPerseveration[:,0] = [[1,0,0,0],[0,0,1,0]]
#     else:
#         qPerseveration = np.zeros((nReps,obj.nTrials,len(stimNames)))
#         if initPerseveration:
#             qPerseveration[:,0] = [1,0,1,0]

#     qReward = np.zeros((nReps,obj.nTrials))

#     qTotal = np.zeros((nReps,obj.nTrials))

#     pAction = np.zeros((nReps,obj.nTrials))
    
#     action = np.zeros((nReps,obj.nTrials),dtype=int)
    
#     for i in range(nReps):
#         for trial,stim in enumerate(obj.trialStim):
#             # if optoLabel is not None and obj.trialOptoLabel[trial] in optoLabel:
#             #     betaAct = betaActionOpto if not np.isnan(betaActionOpto) else betaAction
#             #     biasAct = biasActionOpto if not np.isnan(biasActionOpto) else biasAction
#             # else:
#             #     betaAct = betaAction
#             #     biasAct = biasAction 

#             if stim != 'catch':
#                 modality = 0 if 'vis' in stim else 1
#                 pStim = np.zeros(len(stimNames))
#                 pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]

#                 if stateSpace:
#                     pState = pContext[i,trial][:,None] * pStim[None,:]
#                 else:
#                     pState = np.repeat(pContext[i,trial],2) * pStim

#                 expectedOutcomeContext = np.sum(pState * qContext)

#                 expectedOutcome = np.sum((pStim if wContext > 0 or np.isnan(alphaContext) else pState) * qReinforcement[i,trial])

#                 expectedAction = np.sum((pState if contextPerseveration else pStim) * qPerseveration[i,trial])

#                 bias = biasAction + qReward[i,trial]

#                 qTotal[i,trial] = (wContext * (2 * (expectedOutcomeContext + bias) - 1)) + (wReinforcement * (2 * (expectedOutcome + bias) - 1)) + (wPerseveration * (2 * (expectedAction + bias) - 1))

#                 pAction[i,trial] = 1 / (1 + np.exp(-qTotal[i,trial]))
                
#                 if useChoiceHistory:
#                     action[i,trial] = obj.trialResponse[trial]
#                 elif random.random() < pAction[i,trial]:
#                     action[i,trial] = 1 
            
#             if trial+1 < obj.nTrials:
#                 pContext[i,trial+1] = pContext[i,trial]
#                 qReinforcement[i,trial+1] = qReinforcement[i,trial]
#                 qPerseveration[i,trial+1] = qPerseveration[i,trial]
#                 qReward[i,trial+1] = qReward[i,trial]
#                 reward = (action[i,trial] and stim == obj.rewardedStim[trial]) or obj.autoRewardScheduled[trial]
                
#                 if stim != 'catch':
#                     if action[i,trial]:
#                         if not np.isnan(alphaContext):
#                             if scalarError:
#                                 contextError = reward - (expectedOutcomeContext if wContext > 0 else expectedOutcome)
#                             elif reward:
#                                 contextError = 1 - pContext[i,trial,modality]
#                             else:
#                                 contextError = -pContext[i,trial,modality] * pStim[(0 if modality==0 else 2)]
#                             pContext[i,trial+1,modality] += contextError * (alphaContextNeg if not np.isnan(alphaContextNeg) and not reward else alphaContext)
#                             pContext[i,trial+1,modality] = np.clip(pContext[i,trial+1,modality],0,1)
                        
#                         if not np.isnan(alphaReinforcement):
#                             if scalarError:
#                                 outcomeError = reward - expectedOutcome
#                             else:
#                                 outcomeError = (pStim if wContext > 0 or np.isnan(alphaContext) else pState) * (reward - qReinforcement[i,trial])
#                             qReinforcement[i,trial+1] += outcomeError * (alphaReinforcementNeg if not np.isnan(alphaReinforcementNeg) and not reward else alphaReinforcement)
#                             qReinforcement[i,trial+1] = np.clip(qReinforcement[i,trial+1],0,1)
            
#                         if not np.isnan(alphaPerseveration):
#                             if scalarError:
#                                 actionError = action[i,trial] - expectedAction
#                             else:
#                                 actionError = (pState if contextPerseveration else pStim) * (action[i,trial] - qPerseveration[i,trial])
#                             qPerseveration[i,trial+1] += actionError * alphaPerseveration
#                             qPerseveration[i,trial+1] = np.clip(qPerseveration[i,trial+1],0,1)
                
#                 iti = obj.stimStartTimes[trial+1] - obj.stimStartTimes[trial]

#                 if not np.isnan(alphaContext):
#                     decay = 0
#                     if not np.isnan(tauContext):
#                         decay += (1 - np.exp(-iti/tauContext)) * (0.5 - pContext[i,trial+1,modality])
#                     if not np.isnan(blockTiming):
#                         blockTime = obj.stimStartTimes[trial+1] - obj.stimStartTimes[np.where(obj.trialBlock==obj.trialBlock[trial])[0][0]]
#                         if blockTime > 600 / blockTimingShape / 2:
#                             blockTimeAmp = (np.cos((2 * np.pi * blockTimingShape * (600 - blockTime)) / 600) + 1) / 2
#                             decay += (blockTiming * blockTimeAmp) * (0.5 - pContext[i,trial+1,modality])
#                     pContext[i,trial+1,modality] += decay
#                     pContext[i,trial+1,(1 if modality==0 else 0)] = 1 - pContext[i,trial+1,modality]

#                 if not np.isnan(tauReinforcement):
#                     qReinforcement[i,trial+1] *= np.exp(-iti/tauReinforcement)

#                 if not np.isnan(tauPerseveration):
#                     qPerseveration[i,trial+1] *= np.exp(-iti/tauPerseveration)

#                 if not np.isnan(alphaReward):
#                     if reward:
#                         qReward[i,trial+1] += alphaReward
#                     qReward[i,trial+1] *= np.exp(-iti/tauReward)
    
#     return pContext, qReinforcement, qPerseveration, qReward, qTotal, pAction, action


def runModel(obj,visConfidence,audConfidence,
             wContext,alphaContext,alphaContextNeg,tauContext,blockTiming,blockTimingShape,
             wReinforcement,alphaReinforcement,alphaReinforcementNeg,tauReinforcement,
             wPerseveration,alphaPerseveration,tauPerseveration,wReward,alphaReward,tauReward,wBias,
             useChoiceHistory=True,nReps=1):

    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    modality = 0

    pContext = 0.5 + np.zeros((nReps,obj.nTrials,2))
    qContext = np.array([1,0,1,0])

    qReinforcement = np.zeros((nReps,obj.nTrials,len(stimNames)))
    qReinforcement[:,0] = [1,0,1,0]

    qPerseveration = np.zeros((nReps,obj.nTrials,len(stimNames)))

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

                pState = pStim * np.repeat(pContext[i,trial],2)

                expectedOutcomeContext = np.sum(pState * qContext)

                expectedOutcome = np.sum(pStim * qReinforcement[i,trial])

                expectedAction = np.sum(pStim * qPerseveration[i,trial])

                qTotal[i,trial] = (wContext * (2*expectedOutcomeContext-1)) + (wReinforcement * (2*expectedOutcome-1)) + (wPerseveration * (2*expectedAction-1)) + (wReward * (2*qReward[i,trial]-1)) + wBias

                pAction[i,trial] = 1 / (1 + np.exp(-qTotal[i,trial]))
                
                if useChoiceHistory:
                    action[i,trial] = obj.trialResponse[trial]
                elif random.random() < pAction[i,trial]:
                    action[i,trial] = 1 
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qReinforcement[i,trial+1] = qReinforcement[i,trial]
                qPerseveration[i,trial+1] = qPerseveration[i,trial]
                qReward[i,trial+1] = qReward[i,trial]
                reward = (action[i,trial] and stim == obj.rewardedStim[trial]) or obj.autoRewardScheduled[trial]
                
                if stim != 'catch':
                    if action[i,trial] or reward:
                        if wContext > 0:
                            if reward:
                                contextError = 1 - pContext[i,trial,modality]
                            else:
                                contextError = -pContext[i,trial,modality] * pStim[(0 if modality==0 else 2)]
                            pContext[i,trial+1,modality] += contextError * (alphaContextNeg if not np.isnan(alphaContextNeg) and not reward else alphaContext)
                        
                        if wReinforcement > 0:
                            outcomeError = pStim * (reward - qReinforcement[i,trial])
                            qReinforcement[i,trial+1] += outcomeError * (alphaReinforcementNeg if not np.isnan(alphaReinforcementNeg) and not reward else alphaReinforcement)
                    
                    if wPerseveration > 0:
                        actionError = pStim * (action[i,trial] - qPerseveration[i,trial])
                        qPerseveration[i,trial+1] += actionError * alphaPerseveration
                
                iti = obj.stimStartTimes[trial+1] - obj.stimStartTimes[trial]

                if not np.isnan(alphaContext):
                    decay = 0
                    if not np.isnan(tauContext):
                        decay += (1 - np.exp(-iti/tauContext)) * (0.5 - pContext[i,trial+1,modality])
                    if not np.isnan(blockTiming):
                        blockTime = obj.stimStartTimes[trial+1] - obj.stimStartTimes[np.where(obj.trialBlock==obj.trialBlock[trial])[0][0]]
                        if blockTime > 600 / blockTimingShape / 2:
                            blockTimeAmp = (np.cos((2 * np.pi * blockTimingShape * (600 - blockTime)) / 600) + 1) / 2
                            decay += (blockTiming * blockTimeAmp) * (0.5 - pContext[i,trial+1,modality])
                    pContext[i,trial+1,modality] += decay
                    pContext[i,trial+1,(1 if modality==0 else 0)] = 1 - pContext[i,trial+1,modality]

                if not np.isnan(tauReinforcement):
                    qReinforcement[i,trial+1] *= np.exp(-iti/tauReinforcement)

                if not np.isnan(tauPerseveration):
                    qPerseveration[i,trial+1] *= np.exp(-iti/tauPerseveration)

                if not np.isnan(alphaReward):
                    if reward:
                        qReward[i,trial+1] += alphaReward
                    qReward[i,trial+1] *= np.exp(-iti/tauReward)
    
    return pContext, qReinforcement, qPerseveration, qReward, qTotal, pAction, action


def insertFixedParamVals(fitParams,fixedInd,fixedVal):
    nParams = len(fitParams) + len(fixedInd)
    params = np.full(nParams,np.nan)
    params[fixedInd] = fixedVal
    params[[i for i in range(nParams) if i not in fixedInd]] = fitParams
    return params


def evalModel(params,*args):
    trainData,trainingPhase,trainDataTrialCluster,clust,fixedInd,fixedVal,paramsDict = args
    if fixedInd is not None:
        params = insertFixedParamVals(params,fixedInd,fixedVal)
    response = np.concatenate([obj.trialResponse for obj in trainData])
    prediction = np.concatenate([runModel(obj,*params,**paramsDict)[-2][0] for obj in trainData])
    sampleWeight = None
    if clust is not None:
        trials = np.concatenate(trainDataTrialCluster) == clust
    elif 'optoLabel' in paramsDict and paramsDict['optoLabel'] is not None:
        trials = np.concatenate([np.in1d(obj.trialOptoLabel,('no opto',)+paramsDict['optoLabel']) for obj in trainData])
    else:
        trials = np.ones(response.size,dtype=bool)
        # sampleWeight = []
        # for obj in trainData:
        #     sampleWeight.append(np.ones(obj.nTrials))
        #     catchTrials = obj.trialStim=='catch'
        #     sampleWeight[-1][catchTrials] = 0
        #     for blockInd in range(6):
        #         blockTrials = np.where(obj.trialBlock==blockInd+1)[0]
        #         for stim in ('vis1','vis2','sound1','sound2'):
        #             stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==stim)[0])
        #             firstTrial = 1 if stimTrials[0]==blockTrials[0] else 0
        #             sampleWeight[-1][stimTrials[firstTrial]] = (obj.nTrials - obj.catchTrials.sum()) / (4 * 6)
        # sampleWeight = np.concatenate(sampleWeight)
    response = response[trials]
    prediction = prediction[trials]
    logLoss = sklearn.metrics.log_loss(response,prediction,normalize=True,sample_weight=sampleWeight)
    return logLoss


def fitModel(mouseId,trainingPhase,testData,trainData,modelType):

    modelParams = {'visConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
                   'audConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
                   'wContext': {'bounds': (0,50), 'fixedVal': 0},
                   'alphaContext': {'bounds':(0,1), 'fixedVal': np.nan},
                   'alphaContextNeg': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauContext': {'bounds': (1,300), 'fixedVal': np.nan},
                   'blockTiming': {'bounds': (0,1), 'fixedVal': np.nan},
                   'blockTimingShape': {'bounds': (0.5,4), 'fixedVal': np.nan},
                   'wReinforcement': {'bounds': (0,50), 'fixedVal': 0},
                   'alphaReinforcement': {'bounds': (0,1), 'fixedVal': np.nan},
                   'alphaReinforcementNeg': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauReinforcement': {'bounds': (1,10000), 'fixedVal': np.nan},
                   'wPerseveration': {'bounds': (0,50), 'fixedVal': 0},
                   'alphaPerseveration': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauPerseveration': {'bounds': (1,300), 'fixedVal': np.nan},
                   'wReward': {'bounds': (0,50), 'fixedVal': 0},
                   'alphaReward': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauReward': {'bounds': (1,60), 'fixedVal': np.nan},
                   'wBias': {'bounds':(0,50), 'fixedVal': 0},}
    modelParamNames = list(modelParams.keys())

    paramsDict = {}
    # paramsDict['optoLabel'] = None
    # if modelType in ('BasicRL','ContextRL','basicRL_learningRates','contextRL_learningRates'):
    #     paramsDict['initReinforcement'] = True
    # else:
    #     for paramsOption in ('stateSpace','contextPerseveration','initReinforcement','initPerseveration','scalarError'):
    #         if paramsOption in modelType:
    #             paramsDict[paramsOption] = True

    if trainingPhase == 'clusters':
        clustData = np.load(os.path.join(baseDir,'Sam','clustData.npy'),allow_pickle=True).item()
        testDataTrialCluster = clustData['trialCluster'][testData.subjectName][testData.startTime]
        trainDataTrialCluster = [clustData['trialCluster'][obj.subjectName][obj.startTime] for obj in trainData]
        clustIds = (3,4,5,6) # np.unique(clustData['clustId'])
    else:
        testDataTrialCluster = None
        trainDataTrialCluster = None
        clustIds = (None,)

    # fitFunc = scipy.optimize.direct
    # fitFuncParams = {'eps': 1e-3,'maxfun': None,'maxiter': int(1e3),'locally_biased': False,'vol_tol': 1e-16,'len_tol': 1e-6}
    fitFunc = scipy.optimize.differential_evolution
    fitFuncParams = {'mutation': (0.5,1),'recombination': 0.7,'popsize': 16,'strategy': 'best1bin', 'init': 'sobol'}

    fileName = str(mouseId)+'_'+testData.startTime+'_'+trainingPhase+'_'+modelType+'.npz'
    filePath = os.path.join(baseDir,'Sam','RLmodel',fileName)

    otherFixedPrms = [[]]
    if modelType == 'BasicRL':
        if trainingPhase == 'clusters':
            otherFixedPrms += []
        else:
            otherFixedPrms += [['alphaReinforcement'],['wPerseveration','alphaPerseveration','tauPerseveration'],
                               ['alphaReward','tauReward']]
        fixedParams = [['wContext','alphaContext','alphaContextNeg','tauContext','blockTiming','blockTimingShape',
                        'alphaReinforcementNeg','tauReinforcement']
                        + prms for prms in otherFixedPrms]
    elif modelType == 'basicRL_learningRates':
        otherFixedPrms += [['alphaReinforcementNeg']]
        fixedParams = [['wContext','alphaContext','alphaContextNeg','tauContext','blockTiming','blockTimingShape','tauReinforcement']
                        + prms for prms in otherFixedPrms]                    
    elif modelType == 'ContextRL':
        if trainingPhase == 'clusters':
            otherFixedPrms += [] 
        elif trainingPhase == 'ephys':
            otherFixedPrms += []
        elif trainingPhase in ('nogo','noAR'):
            otherFixedPrms += []
        else:
            otherFixedPrms += [['tauContext'],['blockTiming','blockTimingShape'],['tauContext','blockTiming','blockTimingShape'],
                               ['alphaReinforcement'],['wPerseveration','alphaPerseveration','tauPerseveration'],
                               ['alphaReward','tauReward']]
        fixedParams = [['wContext','alphaContextNeg','blockTiming','blockTimingShape','alphaReinforcementNeg','tauReinforcement']
                        + prms for prms in otherFixedPrms]
    elif modelType == 'contextRL_learningRates':
        otherFixedPrms += [['alphaContextNeg'],['alphaReinforcementNeg'],['alphaContextNeg','alphaReinforcementNeg']]
        fixedParams = [['wContext','tauReinforcement']
                        + prms for prms in otherFixedPrms]
    elif modelType == 'MixedAgentRL':
        otherFixedPrms += [['wContext','alphaContext','tauContext','wReinforcement','alphaReinforcement'],
                           ['wContext','alphaContext','tauContext'],
                           ['wReinforcement','alphaReinforcement'],
                           ['wPerseveration','alphaPerseveration','tauPerseveration'],
                           ['wReward','alphaReward','tauReward'],
                           ['wBias'],
                           ['tauContext']]
        fixedParams = [['alphaContextNeg','blockTiming','blockTimingShape','alphaReinforcementNeg','tauReinforcement']
                        + prms for prms in otherFixedPrms]
    elif 'multiAgent' in modelType:
        fixedParams = [['alphaContextNeg','blockTiming','blockTimingShape','alphaReinforcementNeg']]
    else:
        fixedParams = [['wContext','alphaContextNeg','blockTiming','blockTimingShape','alphaReinforcementNeg']]
    
    params = []
    logLoss = []
    terminationMessage = []
    for fixedPrms in fixedParams:
        fixedParamIndices = [modelParamNames.index(prm) for prm in fixedPrms]
        fixedParamValues = [modelParams[prm]['fixedVal'] for prm in fixedPrms]
        bounds = tuple(modelParams[prm]['bounds'] for prm in modelParamNames if prm not in fixedPrms)
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
            if clust is None:
                trData = trainData
                trClust = trainDataTrialCluster
            else:
                trainSessionsWithClust = [(obj,trialCluster) for obj,trialCluster in zip(trainData,trainDataTrialCluster) if np.any(trialCluster==clust)]
                if np.any(testDataTrialCluster==clust) and len(trainSessionsWithClust) > 0:
                    maxTrainSessions = 10
                    if len(trainSessionsWithClust) > maxTrainSessions:
                        trainSessionsWithClust = random.sample(trainSessionsWithClust,maxTrainSessions)
                    trData,trClust = zip(*trainSessionsWithClust)
                else:
                    prms.append(np.full(len(modelParams),np.nan))
                    nll.append(np.nan)
                    tm.append('')
                    continue
            
            fit = fitFunc(evalModel,bounds,args=(trData,trainingPhase,trClust,clust,fixedParamIndices,fixedParamValues,paramsDict),**fitFuncParams)
            prms.append(insertFixedParamVals(fit.x,fixedParamIndices,fixedParamValues))
            nll.append(fit.fun)
            tm.append(fit.message)

    np.savez(filePath,params=params,logLoss=logLoss,terminationMessage=terminationMessage,
             trainSessions=[obj.startTime for obj in trainData],**paramsDict) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=int)
    parser.add_argument('--sessionIndex',type=int)
    parser.add_argument('--trainingPhase',type=str)
    parser.add_argument('--modelType',type=str)
    args = parser.parse_args()
    trainingPhase = args.trainingPhase.replace('_',' ')
    testData,trainData = getSessionsToFit(args.mouseId,trainingPhase,args.sessionIndex)
    fitModel(args.mouseId,trainingPhase,testData,trainData,args.modelType)
