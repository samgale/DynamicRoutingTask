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
        testSession = sessions[sessionIndex]
        trainSessions = [s for s in sessions if s != testSession]
    else:
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
            # trainSessions = preExperimentSessions[-4:]
            trainSessions = [s for s in sessions if s != testSession]
    testData = getSessionData(mouseId,df.loc[testSession,'start time'])
    trainData = [getSessionData(mouseId,startTime) for startTime in df.loc[trainSessions,'start time']]
    return testData,trainData


def calcLogisticProb(q,beta,bias,lapse):
    return (1 - lapse) / (1 + np.exp(-beta * (q - 0.5 + bias)))


def runModel(obj,betaAction,biasAction,lapseRate,biasAttention,visConfidence,audConfidence,
             wContext,alphaContext,alphaContextNeg,decayContext,blockTiming,blockTimingShape,
             alphaReinforcement,alphaReinforcementNeg,alphaUncertainty,
             rewardBias,rewardBiasTau,noRewardBias,noRewardBiasTau,perseverationBias,perseverationTau,
             betaActionOpto,biasActionOpto,wContextOpto,optoLabel=None,useChoiceHistory=True,nReps=1):

    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    modality = 0

    wCntx = 0.5 + np.zeros((nReps,obj.nTrials))
    qUncertainty = np.ones((nReps,obj.nTrials))

    pContext = 0.5 + np.zeros((nReps,obj.nTrials,2))
    qContext = np.array([visConfidence,1-visConfidence,audConfidence,1-audConfidence])

    qReinforcement = np.zeros((nReps,obj.nTrials,len(stimNames)))
    qReinforcement[:,0] = [visConfidence,1-visConfidence,audConfidence,1-audConfidence]

    qReward = np.zeros((nReps,obj.nTrials))

    qNoReward = np.zeros((nReps,obj.nTrials))

    qPerseveration = np.zeros((nReps,obj.nTrials))

    qTotal = np.zeros((nReps,obj.nTrials))

    pAction = np.zeros((nReps,obj.nTrials))
    
    action = np.zeros((nReps,obj.nTrials),dtype=int)
    
    for i in range(nReps):
        lastRewardTime = 0
        for trial,stim in enumerate(obj.trialStim):
            if optoLabel is not None and obj.trialOptoLabel[trial] in optoLabel:
                betaAct = betaActionOpto if betaActionOpto is not None else betaAction
                biasAct = biasActionOpto if biasActionOpto is not None else biasAction
            else:
                betaAct = betaAction
                biasAct = biasAction 

            if stim != 'catch':
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]
                if biasAttention > 0:
                    pStim[-2:] *= 1 - biasAttention
                else:
                    pStim[:2] *= 1 + biasAttention

                if not np.isnan(wContext):
                    expectedValue = ((wCntx[i,trial] * np.sum(qContext * pStim * np.repeat(pContext[i,trial],2))) + 
                                     ((1-wCntx[i,trial]) * np.sum(qReinforcement[i,trial] * pStim)))
                elif not np.isnan(alphaContext):
                    expectedValue = np.sum(qReinforcement[i,trial] * pStim * np.repeat(pContext[i,trial],2))
                else:
                    expectedValue = np.sum(qReinforcement[i,trial] * pStim)

                qTotal[i,trial] = expectedValue
                qTotal[i,trial] += qReward[i,trial] + qNoReward[i,trial] + qPerseveration[i,trial]

                pAction[i,trial] = calcLogisticProb(qTotal[i,trial],betaAct,biasAct,lapseRate)
                
                if useChoiceHistory:
                    action[i,trial] = obj.trialResponse[trial]
                elif random.random() < pAction[i,trial]:
                    action[i,trial] = 1 
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qReinforcement[i,trial+1] = qReinforcement[i,trial]
                qUncertainty[i,trial+1] = qUncertainty[i,trial]
                qReward[i,trial+1] = qReward[i,trial]
                qNoReward[i,trial+1] = qNoReward[i,trial]
                qPerseveration[i,trial+1] = qPerseveration[i,trial]
                
                outcome = (action[i,trial] and stim == obj.rewardedStim[trial]) or obj.autoRewardScheduled[trial]
                resp = action[i,trial] or obj.autoRewardScheduled[trial]
                if outcome:
                    lastRewardTime = obj.stimStartTimes[trial]
                
                if stim != 'catch' and resp:
                    if not np.isnan(alphaContext):
                        if outcome:
                            contextError = 1 - pContext[i,trial,modality]
                        else:
                            contextError = -pContext[i,trial,modality] * pStim[(0 if modality==0 else 2)]
                        pContext[i,trial+1,modality] += contextError * (alphaContextNeg if not np.isnan(alphaContextNeg) and contextError < 0 else alphaContext)
                        pContext[i,trial+1,modality] = np.clip(pContext[i,trial+1,modality],0,1)
                    
                    predictionError = pStim * (outcome - qReinforcement[i,trial])
                    if np.isnan(wContext) and not np.isnan(alphaContext):
                        predictionError *= np.repeat(pContext[i,trial],2)
                    if alphaReinforcement > 0:
                        if np.isnan(alphaReinforcementNeg):
                            alphaR = alphaReinforcement
                        else:
                            alphaR = np.zeros(len(stimNames))
                            alphaR[predictionError < 0] = alphaReinforcementNeg
                            alphaR[predictionError > 0] = alphaReinforcement
                        qReinforcement[i,trial+1] += predictionError * alphaR
                        qReinforcement[i,trial+1] = np.clip(qReinforcement[i,trial+1],0,1)
                    if alphaUncertainty > 0:
                        qUncertainty[i,trial+1] += alphaUncertainty * (abs(np.sum(predictionError)) - qUncertainty[i,trial])
                        qUncertainty[i,trial+1] = np.clip(qUncertainty[i,trial+1],0,1)
            
                    if perseverationBias > 0:
                        qPerseveration[i,trial+1] += perseverationBias
                
                iti = obj.stimStartTimes[trial+1] - obj.stimStartTimes[trial]

                decay = 0
                if not np.isnan(decayContext):
                    decay += (1 - np.exp(-iti/decayContext)) * (0.5 - pContext[i,trial+1,modality])
                if not np.isnan(blockTiming):
                    blockTime = obj.stimStartTimes[trial+1] - obj.stimStartTimes[np.where(obj.trialBlock==obj.trialBlock[trial])[0][0]]
                    if blockTime > 600 / blockTimingShape / 2:
                        blockTimeAmp = (np.cos((2 * np.pi * blockTimingShape * (600 - blockTime)) / 600) + 1) / 2
                        decay += (blockTiming * blockTimeAmp) * (0.5 - pContext[i,trial+1,modality])
                pContext[i,trial+1,modality] += decay
                pContext[i,trial+1,(1 if modality==0 else 0)] = 1 - pContext[i,trial+1,modality]

                if not np.isnan(wContext):
                    contextCertainty = 2 * (pContext[i,trial+1].max() - 0.5)
                    wCntx[i,trial+1] = contextCertainty / (contextCertainty + (1 - qUncertainty[i,trial+1]))

                if rewardBias > 0:
                    if outcome > 0:
                        qReward[i,trial+1] += rewardBias
                    qReward[i,trial+1] *= np.exp(-iti/rewardBiasTau)

                if noRewardBias > 0:
                    if outcome > 0:
                        qNoReward[i,trial+1] = 0
                    else:
                        qNoReward[i,trial+1] = noRewardBias * np.exp((obj.stimStartTimes[trial+1] - lastRewardTime)/noRewardBiasTau)

                if perseverationBias > 0:
                    qPerseveration[i,trial+1] *= np.exp(-iti/perseverationTau)
    
    return pContext, qReinforcement, qReward, qTotal, pAction, action


def insertFixedParamVals(fitParams,fixedInd,fixedVal):
    nParams = len(fitParams) + len(fixedInd)
    params = np.full(nParams,np.nan)
    params[fixedInd] = fixedVal
    params[[i for i in range(nParams) if i not in fixedInd]] = fitParams
    return params


def calcPrior(params):
    delta = 0.025
    p = 1
    for i,val in enumerate(params):
        if i in (5,8,10,12,14):
            f = scipy.stats.norm(0,0.5).cdf
            p *= f(val+delta) - f(val-delta)
        elif i in (6,9,11,13):
            f = scipy.stats.beta(2,2).cdf
            p *= f(val+delta) - f(val-delta)
    return p


def getModelRegressors(modelType,modelTypeDict,params,sessions):
    regressors = ['context','reinforcement','reward','bias']
    x = {r: [] for r in regressors}
    y = []
    sessionTrials = []
    for obj in sessions:
        for reg in regressors[:-1]:
            betaAction,biasAction,biasAttention,visConfidence,audConfidence,wContext,alphaContext,decayContext,alphaReinforcement,wReward,alphaReward,wPerseveration,alphaPerseveration = params
            if reg == 'context':
                wContext = 1
                wReward = 0
            elif reg == 'reinforcement':
                wContext = 0
                alphaContext = 0
                wReward = 0
            elif reg == 'reward':
                wContext = 0
                alphaContext = 0
                wReward = 1
            params = (betaAction,biasAction,biasAttention,visConfidence,audConfidence,wContext,alphaContext,decayContext,alphaReinforcement,wReward,alphaReward,wPerseveration,alphaPerseveration)
            x[reg].append(runModel(obj,*params,**modelTypeDict)[-2][0])
        x['bias'].append(np.ones(obj.nTrials))
        y.append(obj.trialResponse)
        sessionTrials.append(obj.nTrials)
    if modelType == 'psytrack':
        d = {'inputs': {key: np.concatenate(val)[:,None] for key,val in x.items()},
             'y': np.concatenate(y).astype(float),
             'dayLength': np.array(sessionTrials)}
        weights = {key: 1 for key in d['inputs']}
        nWeights = sum(weights.values())
        hyper= {'sigInit': 2**4.,
                'sigma': [2**-4.] * nWeights,
                'sigDay': [2**-4.] * nWeights}
        optList = ['sigma','sigDay']
        return d,weights,hyper,optList
    elif modelType == 'glmhmm':
        # list of ntrials x nregressors array for each session
        inputs = [np.stack([x[reg][i] for reg in regressors],axis=-1) for i in range(len(y))]
        resp = [a[:,None].astype(int) for a in y]
        return inputs,resp


def evalModel(params,*args):
    trainData,trainDataTrialCluster,clust,fixedInd,fixedVal,modelType,modelTypeDict = args
    if fixedInd is not None:
        params = insertFixedParamVals(params,fixedInd,fixedVal)
    if modelType == 'psytrack':
        d,weights,hyper,optList = getModelRegressors(modelType,modelTypeDict,params,trainData)
        try:
            hyp,evd,wMode,hessInfo = psytrack.hyperOpt(d,hyper,weights,optList)
            return -evd
        except:
            return 1e6
    elif modelType == 'glmhmm':
        nCategories = 2 # binary choice (go/nogo)
        obsDim = 1 # number of observed dimensions (choice)
        inputDim = 4 # input dimensions
        nStates = 3
        # list of ntrials x nregressors array for each session
        inputs,resp = getModelRegressors(modelType,modelTypeDict,params,trainData)
        glmhmm = ssm.HMM(nStates,obsDim,inputDim,observations="input_driven_obs",observation_kwargs=dict(C=nCategories),transitions="standard")
        fitLL = glmhmm.fit(resp,inputs,method="em",num_iters=200,tolerance=10**-4)
        return -fitLL[-1]
    else:
        response = np.concatenate([obj.trialResponse for obj in trainData])
        prediction = np.concatenate([runModel(obj,*params,**modelTypeDict)[-2][0] for obj in trainData])
        if clust is not None:
            clustTrials = np.concatenate(trainDataTrialCluster) == clust
            response = response[clustTrials]
            prediction = prediction[clustTrials]
        elif 'optoLabel' in modelTypeDict and modelTypeDict['optoLabel'] is not None:
            trials = np.concatenate([np.in1d(obj.trialOptoLabel,('no opto',)+modelTypeDict['optoLabel']) for obj in trainData])
            response = response[trials]
            prediction = prediction[trials]
        logLoss = sklearn.metrics.log_loss(response,prediction)
        # logLoss += -np.log(calcPrior(params))
        return logLoss


def fitModel(mouseId,trainingPhase,testData,trainData):

    modelParams = {'betaAction': {'bounds': (1,40), 'fixedVal': np.nan},
                   'biasAction': {'bounds': (-1,1), 'fixedVal': 0},
                   'lapseRate': {'bounds': (0,1), 'fixedVal': 0},
                   'biasAttention': {'bounds': (-1,1), 'fixedVal': 0},
                   'visConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
                   'audConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
                   'wContext': {'bounds': (0,1), 'fixedVal': np.nan},
                   'alphaContext': {'bounds':(0,1), 'fixedVal': np.nan},
                   'alphaContextNeg': {'bounds': (0,1), 'fixedVal': np.nan},
                   'decayContext': {'bounds': (10,300), 'fixedVal': np.nan},
                   'blockTiming': {'bounds': (0,1), 'fixedVal': np.nan},
                   'blockTimingShape': {'bounds': (0.5,4), 'fixedVal': np.nan},
                   'alphaReinforcement': {'bounds': (0,1), 'fixedVal': 0},
                   'alphaReinforcementNeg': {'bounds': (0,1), 'fixedVal': np.nan},
                   'alphaUncertainty': {'bounds': (0,1), 'fixedVal': 0},
                   'rewardBias': {'bounds': (0,1), 'fixedVal': 0},
                   'rewardBiasTau': {'bounds': (1,50), 'fixedVal': np.nan},
                   'noRewardBias': {'bounds': (0,1), 'fixedVal': 0},
                   'noRewardBiasTau': {'bounds': (10,300), 'fixedVal': np.nan},
                   'perseverationBias': {'bounds': (0,1), 'fixedVal': 0},
                   'perseverationTau': {'bounds': (1,300), 'fixedVal': np.nan},
                   'betaActionOpto': {'bounds': (1,40), 'fixedVal': np.nan},
                   'biasActionOpto': {'bounds': (-1,1), 'fixedVal': np.nan},
                   'wContextOpto': {'bounds': (0,1), 'fixedVal': np.nan}}
    modelParamNames = list(modelParams.keys())

    modelTypeParams = ('optoLabel',)
    modelTypes,modelTypeParamVals = zip(
                                        #('basicRL', (None,)),
                                        ('contextRL', (None,)),
                                        #('mixedAgentRL', (None,)),
                                        #('perseverativeRL', (None,)),
                                        #('psytrack', (None,)),
                                        #('glmhmm', (None,)),
                                        #('contextRLOpto', (('lFC','PFC'),)),
                                        #('mixedAgentRLOpto', (('lFC','PFC'),)),
                                       )

    if trainingPhase == 'clusters':
        clustData = np.load(os.path.join(baseDir,'Sam','clustData.npy'),allow_pickle=True).item()
        testDataTrialCluster = clustData['trialCluster'][testData.subjectName][testData.startTime]
        trainDataTrialCluster = [clustData['trialCluster'][obj.subjectName][obj.startTime] for obj in trainData]
        clustIds = (3,4,5,6) # np.unique(clustData['clustId'])
    else:
        testDataTrialCluster = None
        trainDataTrialCluster = None
        clustIds = (None,)

    # fitFuncParams = {'eps': 1e-3,'maxfun': None,'maxiter': int(1e3),'locally_biased': False,'vol_tol': 1e-16,'len_tol': 1e-6}
    fitFuncParams = {'mutation': (0.5,1),'recombination': 0.7,'popsize': 15,'strategy': 'best1bin'}

    for modelType,modelTypeVals in zip(modelTypes,modelTypeParamVals):
        fileName = str(mouseId)+'_'+testData.startTime+'_'+trainingPhase+'_'+modelType+'.npz'
        if trainingPhase == 'opto':
            filePath = os.path.join(baseDir,'Sam','RLmodel','opto',fileName)
        elif trainingPhase == 'clusters':
            filePath = os.path.join(baseDir,'Sam','RLmodel','clusters',fileName)
        else:
            filePath = os.path.join(baseDir,'Sam','RLmodel',fileName)
        if os.path.exists(filePath):
            continue

        if modelType == 'basicRL':
            if trainingPhase == 'clusters':
                otherFixedPrms = [[],['alphaReinforcement']]
            else:
                otherFixedPrms = [[]]
            fixedParams = [['lapseRate','wContext','alphaContext','alphaContextNeg','decayContext','blockTiming','blockTimingShape','alphaReinforcementNeg','alphaUncertainty',
                            'noRewardBias','noRewardBiasTau','perseverationBias','perseverationTau','betaActionOpto','biasActionOpto','wContextOpto'] +
                            prms for prms in otherFixedPrms]
        elif modelType == 'contextRL':
            if trainingPhase == 'clusters':
                otherFixedPrms = [[],['decayContext'],['blockTiming','blockTimingShape'],['decayContext','blockTiming','blockTimingShape']]
            elif trainingPhase == 'noAR':
                otherFixedPrms = [[],['alphaContextNeg'],['alphaReinforcementNeg'],['alphaContextNeg','alphaReinforcementNeg']]
            else:
                otherFixedPrms = [[],['decayContext'],['blockTiming','blockTimingShape'],['decayContext','blockTiming','blockTimingShape']]
            fixedParams = [['lapseRate','biasAttention','wContext','alphaUncertainty','noRewardBias','noRewardBiasTau','perseverationBias','perseverationTau',
                            'betaActionOpto','biasActionOpto','wContextOpto'] +
                            prms for prms in otherFixedPrms]
        elif modelType == 'mixedAgentRL':
            fixedParams = [['noRewardBias','noRewardBiasTau','perseverationBias','perseverationTau',
                            'betaActionOpto','biasActionOpto','wContextOpto'] +
                            prms for prms in ([],)]
        modelTypeDict = {p: v for p,v in zip(modelTypeParams,modelTypeVals)}
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
                        trData,trClust = zip(*trainSessionsWithClust)
                    else:
                        prms.append(np.full(len(modelParams),np.nan))
                        nll.append(np.nan)
                        tm.append('')
                        continue
                
                # fit with direct or differential_evolution
                fit = scipy.optimize.differential_evolution(evalModel,bounds,args=(trData,trClust,clust,fixedParamIndices,fixedParamValues,modelType,modelTypeDict),**fitFuncParams)
                prms.append(insertFixedParamVals(fit.x,fixedParamIndices,fixedParamValues))
                nll.append(fit.fun)
                tm.append(fit.message)

        np.savez(filePath,params=params,logLoss=logLoss,terminationMessage=terminationMessage,
                 trainSessions=[obj.startTime for obj in trainData],**modelTypeDict) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=int)
    parser.add_argument('--sessionIndex',type=int)
    parser.add_argument('--trainingPhase',type=str)
    args = parser.parse_args()
    trainingPhase = args.trainingPhase.replace('_',' ')
    testData,trainData = getSessionsToFit(args.mouseId,trainingPhase,args.sessionIndex)
    fitModel(args.mouseId,trainingPhase,testData,trainData)
