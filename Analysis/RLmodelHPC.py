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
        if trainingPhase in ('initial training','early learning','late learning','after learning','learning weights','clusters','cluster weights'):
            preExperimentSessions = np.array(['stage 5' in task for task in df['task version']]) & ~ignore
            firstExperimentSession = getFirstExperimentSession(df)
            if firstExperimentSession is not None:
                preExperimentSessions[firstExperimentSession:] = False
            preExperimentSessions = np.where(preExperimentSessions)[0]
            sessionsToPass = getSessionsToPass(mouseId,df,preExperimentSessions,stage=5)
            learnOnset = np.load(os.path.join(baseDir,'Sam','learnOnset.npy'),allow_pickle=True).item()[mouseId]
            nSessionsToFit = 2
            if trainingPhase == 'initial training':
                sessions = preExperimentSessions[:nSessionsToFit]
            elif trainingPhase == 'early learning':
                sessions = preExperimentSessions[learnOnset+1:learnOnset+1+nSessionsToFit]
            elif trainingPhase == 'late learning':
                sessions = preExperimentSessions[sessionsToPass-2-nSessionsToFit:sessionsToPass-2]
            elif trainingPhase == 'after learning':
                sessions = preExperimentSessions[sessionsToPass:sessionsToPass+nSessionsToFit]
            elif trainingPhase == 'learning weights':
                sessions = np.concatenate((preExperimentSessions[:nSessionsToFit],
                                           preExperimentSessions[learnOnset+1:learnOnset+1+nSessionsToFit],
                                           preExperimentSessions[sessionsToPass-2-nSessionsToFit:sessionsToPass-2],
                                           preExperimentSessions[sessionsToPass:sessionsToPass+nSessionsToFit]))
            elif trainingPhase in ('clusters','cluster weights'):
                sessions = preExperimentSessions[sessionsToPass:sessionsToPass+10]
        elif trainingPhase == 'ephys':
            ephys = np.where(df['ephys'] & ~ignore)[0]
            hab = np.where(df['hab'] & ~ignore)[0][-2:]
            sessions = np.concatenate((ephys,hab))
        else:
            sessions = np.array([trainingPhase in task for task in df['task version']]) & ~ignore
            sessions = np.where(sessions)[0]
    if trainingPhase == 'learning weights':
        testData = None
        trainSessions = sessions
    else:
        testSession = sessions[sessionIndex]
        testData = getSessionData(mouseId,df.loc[testSession,'start time'])
        trainSessions = [s for s in sessions if s != testSession]
    trainData = [getSessionData(mouseId,startTime) for startTime in df.loc[trainSessions,'start time']]
    return testData,trainData


def runModel(obj,visConfidence,audConfidence,
             wContext,alphaContext,alphaContextNeg,tauContext,blockTiming,blockTimingShape,
             wReinforcement,alphaReinforcement,alphaReinforcementNeg,tauReinforcement,
             wPerseveration,alphaPerseveration,tauPerseveration,wReward,alphaReward,tauReward,wBias,
             noAgent=[],useChoiceHistory=True,nReps=1):

    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    modality = 0

    pContext = 0.5 + np.zeros((nReps,obj.nTrials,2))
    qContext = np.array([1,0,1,0])

    qReinforcement = np.zeros((nReps,obj.nTrials,len(stimNames)))
    qReinforcement[:,0] = [0.5,0,0.5,0]

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

                expectedOutcomeContext = 0 if 'context' in noAgent else np.sum(pState * qContext)

                expectedOutcome = 0 if 'reinforcement' in noAgent else np.sum(pStim * qReinforcement[i,trial])

                expectedAction = 0 if 'perseveration' in noAgent else np.sum(pStim * qPerseveration[i,trial])

                rewardMotivation = 0 if 'reward' in noAgent else qReward[i,trial]

                qTotal[i,trial] = (wContext * (2*expectedOutcomeContext-1)) + (wReinforcement * (2*expectedOutcome-1)) + (wPerseveration * (2*expectedAction-1)) + (wReward * (2*rewardMotivation-1)) + wBias

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
                        if not np.isnan(alphaContext):
                            if reward:
                                contextError = 1 - pContext[i,trial,modality]
                            else:
                                contextError = -pContext[i,trial,modality] * pStim[(0 if modality==0 else 2)]
                            pContext[i,trial+1,modality] += contextError * (alphaContextNeg if not np.isnan(alphaContextNeg) and not reward else alphaContext)
                        
                        if not np.isnan(alphaReinforcement):
                            outcomeError = pStim * (reward - qReinforcement[i,trial])
                            qReinforcement[i,trial+1] += outcomeError * (alphaReinforcementNeg if not np.isnan(alphaReinforcementNeg) and not reward else alphaReinforcement)
                    
                    if not np.isnan(alphaPerseveration):
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
                        qReward[i,trial+1] += (1 - qReward[i,trial]) * alphaReward
                    qReward[i,trial+1] *= np.exp(-iti/tauReward)
    
    return pContext, qReinforcement, qPerseveration, qReward, qTotal, pAction, action


def insertFixedParamVals(fitParams,fixedInd,fixedVal):
    nParams = len(fitParams) + len(fixedInd)
    params = np.full(nParams,np.nan)
    params[fixedInd] = fixedVal
    params[[i for i in range(nParams) if i not in fixedInd]] = fitParams
    return params


def calcPrior(params,paramNames):
    p = 1
    for prm,val in zip(paramNames,params):
        if any([w in prm for w in ('wContext','wReinforcement','wPerseveration','wReward')]) and val > 0:
            p *= scipy.stats.norm(0,10).pdf(val)
    return p


def evalModel(params,*args):
    trainData,trainTrials,trainingPhase,fixedInd,fixedVal,paramNames,paramsDict,clustIds,trainTrialClust = args
    if fixedInd is not None:
        params = insertFixedParamVals(params,fixedInd,fixedVal)

    response = np.concatenate([obj.trialResponse for obj in trainData])
    if trainingPhase in ('clusters','cluster weights'):
        prediction = np.full(trainTrialClust.size,np.nan)
        for i,clust in enumerate(clustIds):
            prms = params[:paramNames.index('wBias')+1].copy()
            if trainingPhase == 'cluster weights' and i > 0:
                for prm in ('wContext','wReinforcement','wPerseveration','wReward','wBias'):
                    prms[paramNames.index(prm)] = params[paramNames.index(prm+str(i))]
            pred = np.concatenate([runModel(obj,*prms,**paramsDict)[-2][0] for obj in trainData])
            isClust = trainTrialClust == clust
            prediction[isClust] = pred[isClust]
        trials = ~np.isnan(prediction)
    elif trainingPhase == 'learning weights':
        prediction = []
        for i in range(4):
            prms = params[:paramNames.index('wBias')+1].copy()
            if i > 0:
                for prm in ('wContext','wReinforcement','wPerseveration','wReward','wBias'):
                    prms[paramNames.index(prm)] = params[paramNames.index(prm+str(i))]
            prediction.append(np.concatenate([runModel(obj,*prms,**paramsDict)[-2][0] for obj in trainData[i*2:i*2+2]]))
        prediction = np.concatenate(prediction)
        trials = np.ones(prediction.size,dtype=bool)
    elif trainTrials is not None:
        prediction = np.concatenate([runModel(obj,*params,**paramsDict)[-2][0] for obj in trainData])
        trials = np.zeros(response.size,dtype=bool)
        trials[trainTrials] = True
    else:
        prediction = np.concatenate([runModel(obj,*params,**paramsDict)[-2][0] for obj in trainData])
        if 'optoLabel' in paramsDict and paramsDict['optoLabel'] is not None:
            trials = np.concatenate([np.in1d(obj.trialOptoLabel,('no opto',)+paramsDict['optoLabel']) for obj in trainData])
        else:
            trials = np.ones(response.size,dtype=bool)
    response = response[trials]
    prediction = prediction[trials]
    usePrior = True
    if usePrior:
        logLoss = sklearn.metrics.log_loss(response,prediction,normalize=False,sample_weight=None)
        logLoss += -np.log(calcPrior(params,paramNames))
    else:
        logLoss = sklearn.metrics.log_loss(response,prediction,normalize=True,sample_weight=None)
    return logLoss


def fitModel(mouseId,trainingPhase,testData,trainData,modelType,fixedParamsIndex):

    modelParams = {'visConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
                   'audConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
                   'wContext': {'bounds': (0,30), 'fixedVal': 0},
                   'alphaContext': {'bounds':(0,1), 'fixedVal': np.nan},
                   'alphaContextNeg': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauContext': {'bounds': (1,300), 'fixedVal': np.nan},
                   'blockTiming': {'bounds': (0,1), 'fixedVal': np.nan},
                   'blockTimingShape': {'bounds': (0.5,4), 'fixedVal': np.nan},
                   'wReinforcement': {'bounds': (0,30), 'fixedVal': 0},
                   'alphaReinforcement': {'bounds': (0,1), 'fixedVal': np.nan},
                   'alphaReinforcementNeg': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauReinforcement': {'bounds': (1,300), 'fixedVal': np.nan},
                   'wPerseveration': {'bounds': (0,30), 'fixedVal': 0},
                   'alphaPerseveration': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauPerseveration': {'bounds': (1,600), 'fixedVal': np.nan},
                   'wReward': {'bounds': (0,30), 'fixedVal': 0},
                   'alphaReward': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauReward': {'bounds': (1,60), 'fixedVal': np.nan},
                   'wBias': {'bounds':(0,30), 'fixedVal': 0},}

    if testData is not None:
        fileName = str(mouseId)+'_'+testData.startTime+'_'+trainingPhase+'_'+modelType+('' if fixedParamsIndex=='None' else '_'+fixedParamsIndex)+'.npz'
    
    if trainingPhase in ('clusters','cluster weights'):
        clustData = np.load(os.path.join(baseDir,'Sam','clustData.npy'),allow_pickle=True).item()
        if testData.subjectName not in clustData['trialCluster'] or testData.startTime not in clustData['trialCluster'][testData.subjectName]:
            print('test session not in cluster data')
            return
        testTrialClust = clustData['trialCluster'][testData.subjectName][testData.startTime]
        trainTrialClust = np.concatenate([clustData['trialCluster'][obj.subjectName][obj.startTime] for obj in trainData])
        clustIds = (3,4,5,6) if trainingPhase == 'clusters' else (3,4,6)
        if not np.any(np.in1d(testTrialClust,clustIds)):
            print('test session does not have any of the clusters')
            return
        if trainingPhase == 'clusters':
            if not np.any(np.in1d(np.unique(testTrialClust),np.unique(trainTrialClust))):
                print('training data does not share any clusters with test session')
                return
        else:
            if not np.all(np.in1d(clustIds,np.unique(trainTrialClust))):
                print('training data does not have all of the clusters')
                return
            for prm in ('wContext','wReinforcement','wPerseveration','wReward','wBias'):
                for i,clust in enumerate(clustIds):
                    if i > 0:
                        modelParams[prm+str(i)] = modelParams[prm]
        filePath = [os.path.join(baseDir,'Sam','RLmodel',trainingPhase,fileName)]
    else:
        clustIds = None
        trainTrialClust = None
        if trainingPhase == 'learning weights':
            for prm in ('wContext','wReinforcement','wPerseveration','wReward','wBias'):
                    for i in range(1,4):
                        modelParams[prm+str(i)] = modelParams[prm]
            filePath = []
            for obj,lbl in zip(trainData,[phase for phase in ('initial training','early learning','late learning','after learning') for _ in range(2)]):
                fileName = str(mouseId)+'_'+obj.startTime+'_'+lbl+'_'+modelType+('' if fixedParamsIndex=='None' else '_'+fixedParamsIndex)+'.npz'
                filePath.append(os.path.join(baseDir,'Sam','RLmodel',trainingPhase,fileName))
        else:
            filePath = [os.path.join(baseDir,'Sam','RLmodel','learning',fileName)]

    modelParamNames = list(modelParams.keys())
    paramsDict = {}

    # fitFunc = scipy.optimize.direct
    # fitFuncParams = {'eps': 1e-3,'maxfun': None,'maxiter': int(1e3),'locally_biased': False,'vol_tol': 1e-16,'len_tol': 1e-6}
    fitFunc = scipy.optimize.differential_evolution
    fitFuncParams = {'mutation': (0.5,1),'recombination': 0.7,'popsize': 20,'strategy': 'best1bin', 'init': 'sobol', 'workers': 10} 

    if modelType == 'BasicRL':
        otherFixedPrms = [['wContext','alphaContext','tauContext'],
                          ['wContext','alphaContext','tauContext','wReinforcement','alphaReinforcement'],
                          ['wContext','alphaContext','tauContext','wPerseveration','alphaPerseveration','tauPerseveration'],
                          ['wContext','alphaContext','tauContext','wReward','alphaReward','tauReward'],
                          ['wContext','alphaContext','tauContext','wBias'],
                          []]
        fixedParams = [['alphaContextNeg','blockTiming','blockTimingShape','alphaReinforcementNeg','tauReinforcement']
                        + prms for prms in otherFixedPrms]
    elif modelType == 'ContextRL':
        if trainingPhase == 'clusters':
            otherFixedPrms = [[],
                              ['wContext','alphaContext','tauContext'],
                              ['wReinforcement','alphaReinforcement'],
                              ['wPerseveration','alphaPerseveration','tauPerseveration']]
        elif trainingPhase in ('learning weights','cluster weights'):
            otherFixedPrms = [[],
                              [prm for prm in modelParamNames if 'wContext' in prm] + ['alphaContext','tauContext'],
                              [prm for prm in modelParamNames if 'wReinforcement' in prm] + ['alphaReinforcement'],
                              [prm for prm in modelParamNames if 'wPerseveration' in prm] + ['alphaPerseveration','tauPerseveration']]
        else:
            otherFixedPrms = [[],
                              ['wContext','alphaContext','tauContext'],
                              ['wReinforcement','alphaReinforcement'],
                              ['wContext','alphaContext','tauContext','wReinforcement','alphaReinforcement'],
                              ['wReward','alphaReward','tauReward'],
                              ['wBias'],
                              ['tauContext']]
        fixedParams = [['alphaContextNeg','blockTiming','blockTimingShape','alphaReinforcementNeg','tauReinforcement','wPerseveration','alphaPerseveration','tauPerseveration']
                        + prms for prms in otherFixedPrms]
    elif modelType == 'contextRL_learningRates':
        otherFixedPrms = [['alphaContextNeg']]
        fixedParams = [['blockTiming','blockTimingShape','wReinforcement','alphaReinforcement','alphaReinforcementNeg','tauReinforcement']
                        + prms for prms in otherFixedPrms]
    
    params = []
    logLossTrain = []
    logLossTest = []
    terminationMessage = []
    trainTrials = None
    for fixedPrms in (fixedParams if fixedParamsIndex=='None' else (fixedParams[int(fixedParamsIndex)],)):
        fixedParamIndices = [modelParamNames.index(prm) for prm in fixedPrms]
        fixedParamValues = [modelParams[prm]['fixedVal'] for prm in fixedPrms]
        bounds = tuple(modelParams[prm]['bounds'] for  prm in modelParamNames if prm not in fixedPrms)
        if trainingPhase == 'clusters':
            for clust in clustIds:
                params.append([])
                logLoss.append([])
                terminationMessage.append([])
                if clust in testTrialClust and clust in trainTrialClust:
                    fit = fitFunc(evalModel,bounds,args=(trainData,trainTrials,trainingPhase,fixedParamIndices,fixedParamValues,modelParamNames,paramsDict,(clust,),trainTrialClust),**fitFuncParams)
                    params[-1].append(insertFixedParamVals(fit.x,fixedParamIndices,fixedParamValues))
                    logLossTrain[-1].append(fit.fun)
                    terminationMessage[-1].append(fit.message)
                else:
                    params[-1].append(np.full(len(modelParamNames),np.nan))
                    logLossTrain[-1].append(np.nan)
                    terminationMessage[-1].append('')
        elif len(trainData) < 2:
            params.append([])
            logLossTrain.append([])
            logLossTest.append([])
            nFolds = 5
            n = round(testData.nTrials / nFolds)
            for _ in range(nFolds):
                shuffleInd = np.random.permutation(testData.nTrials)
                prediction = np.full(testData.nTrials,np.nan)
                for k in range(nFolds):
                    start = k * n
                    testTrials = shuffleInd[start:start+n] if k+1 < nFolds else shuffleInd[start:]
                    trainTrials = np.setdiff1d(shuffleInd,testTrials)
                    fit = fitFunc(evalModel,bounds,args=([testData],trainTrials,trainingPhase,fixedParamIndices,fixedParamValues,modelParamNames,paramsDict,clustIds,trainTrialClust),**fitFuncParams)
                    params[-1].append(insertFixedParamVals(fit.x,fixedParamIndices,fixedParamValues))
                    logLossTrain[-1].append(fit.fun)
                    prediction[testTrials] = runModel(testData,*params[-1][-1],**paramsDict)[-2][0][testTrials]
                logLossTest[-1].append(sklearn.metrics.log_loss(testData.trialResponse,prediction,normalize=True))
        else:
            fit = fitFunc(evalModel,bounds,args=(trainData,trainTrials,trainingPhase,fixedParamIndices,fixedParamValues,modelParamNames,paramsDict,clustIds,trainTrialClust),**fitFuncParams)
            params.append(insertFixedParamVals(fit.x,fixedParamIndices,fixedParamValues))
            logLossTrain.append(fit.fun)
            terminationMessage.append(fit.message)

    for f in filePath:
        np.savez(f,params=params,logLossTrain=logLossTrain,logLossTest=logLossTest,terminationMessage=terminationMessage,
                 trainSessions=[obj.startTime for obj in trainData],**paramsDict) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=int)
    parser.add_argument('--sessionIndex',type=int)
    parser.add_argument('--trainingPhase',type=str)
    parser.add_argument('--modelType',type=str)
    parser.add_argument('--fixedParamsIndex',type=str)
    args = parser.parse_args()
    trainingPhase = args.trainingPhase.replace('_',' ')
    testData,trainData = getSessionsToFit(args.mouseId,trainingPhase,args.sessionIndex)
    fitModel(args.mouseId,trainingPhase,testData,trainData,args.modelType,args.fixedParamsIndex)
