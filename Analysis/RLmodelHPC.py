# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import argparse
import os
import pathlib
import random
import numpy as np
import pandas as pd
import scipy
import sklearn.metrics
from DynamicRoutingAnalysisUtils import getSessionData


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting')


def getRandomDrift(nReps,nTrials,sigma=5):
    edgeSamples = 10 * sigma
    nSamples = nTrials + edgeSamples
    drift = np.array([scipy.ndimage.gaussian_filter(np.random.choice((-1,1),nSamples).astype(float),sigma)[edgeSamples:edgeSamples+nTrials] for _ in range(nReps)])
    drift /= np.max(np.absolute(drift),axis=1)[:,None]
    drift += 1
    return drift


def runModel(obj,visConfidence,audConfidence,qInitVis,qInitAud,
             wContext,alphaContext,alphaContextNeg,tauContext,alphaContextReinforcement,
             wReinforcement,alphaReinforcement,alphaReinforcementNeg,tauReinforcement,
             wPerseveration,alphaPerseveration,tauPerseveration,wResponse,alphaResponse,tauResponse,
             wReward,alphaReward,tauReward,wBias,
             noAgent=[],drift=None,useChoiceHistory=True,nReps=1):

    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    modality = 0

    pContext = 0.5 + np.zeros((nReps,obj.nTrials,2))

    qContext = np.zeros((nReps,obj.nTrials,2,len(stimNames)))
    qContext[:,0,0] = [1,0,qInitAud,0]
    qContext[:,0,1] = [qInitVis,0,1,0]
    
    qReinforcement = np.zeros((nReps,obj.nTrials,len(stimNames)))
    qReinforcement[:,0] = [1,0,1,0]

    qPerseveration = np.zeros((nReps,obj.nTrials,len(stimNames)))

    qResponse = np.zeros((nReps,obj.nTrials))

    qReward = np.zeros((nReps,obj.nTrials))

    qTotal = np.zeros((nReps,obj.nTrials))

    pAction = np.zeros((nReps,obj.nTrials))
    
    action = np.zeros((nReps,obj.nTrials),dtype=int)
    
    if drift is not None:
        rd = getRandomDrift(nReps,obj.nTrials)
    
    if drift == 'context drift':
        wContext = wContext * rd
    else:
        wContext = wContext * np.ones((nReps,obj.nTrials))
        
    if drift == 'reinforcement drift':
        wReinforcement = wReinforcement * rd
    else:
        wReinforcement = wReinforcement * np.ones((nReps,obj.nTrials))
    
    if drift == 'balanced drift':
        if wContext[0,0] > wReinforcement[0,0]:
            w = wReinforcement
            wReinforcement = w * rd
            wContext += w - wReinforcement
        else:
            w = wContext
            wContext = w * rd
            wReinforcement += w - wContext

    if drift == 'bias drift':
        wBias = wBias * rd
    else:
        wBias = wBias * np.ones((nReps,obj.nTrials))
    
    for i in range(nReps):
        for trial,stim in enumerate(obj.trialStim):
            if stim != 'catch':
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]

                pState = pStim[None,:] * pContext[i,trial][:,None]

                expectedOutcomeContext = 0 if 'context' in noAgent else np.sum(pState * qContext[i,trial])

                expectedOutcome = 0 if 'reinforcement' in noAgent else np.sum(pStim * qReinforcement[i,trial])

                expectedAction = 0 if 'perseveration' in noAgent else np.sum(pStim * qPerseveration[i,trial])

                qResp = 0 if 'response' in noAgent else qResponse[i,trial]

                qRew = 0 if 'reward' in noAgent else qReward[i,trial]

                qTotal[i,trial] = ((wContext[i,trial] * (2*expectedOutcomeContext-1)) + 
                                   (wReinforcement[i,trial] * (2*expectedOutcome-1)) + 
                                   (wPerseveration * (2*expectedAction-1)) + 
                                   (wResponse * (2*qResp-1)) + 
                                   (wReward * (2*qRew-1)) + 
                                   wBias[i,trial])

                pAction[i,trial] = 1 / (1 + np.exp(-qTotal[i,trial]))
                
                if useChoiceHistory:
                    action[i,trial] = obj.trialResponse[trial]
                elif random.random() < pAction[i,trial]:
                    action[i,trial] = 1 
            
            if trial+1 < obj.nTrials:
                pContext[i,trial+1] = pContext[i,trial]
                qContext[i,trial+1] = qContext[i,trial]
                qReinforcement[i,trial+1] = qReinforcement[i,trial]
                qPerseveration[i,trial+1] = qPerseveration[i,trial]
                qResponse[i,trial+1] = qResponse[i,trial]
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

                        if not np.isnan(alphaContextReinforcement):
                            outcomeError = pState * (reward - expectedOutcomeContext)
                            qContext[i,trial+1] += outcomeError * alphaContextReinforcement
                        
                        if not np.isnan(alphaReinforcement):
                            outcomeError = pStim * (reward - expectedOutcome)
                            qReinforcement[i,trial+1] += outcomeError * (alphaReinforcementNeg if not np.isnan(alphaReinforcementNeg) and not reward else alphaReinforcement)
                    
                    if not np.isnan(alphaPerseveration):
                        actionError = pStim * (action[i,trial] - qPerseveration[i,trial])
                        qPerseveration[i,trial+1] += actionError * alphaPerseveration
                
                iti = obj.stimStartTimes[trial+1] - obj.stimStartTimes[trial]

                if not np.isnan(alphaContext):
                    if not np.isnan(tauContext):
                        pContext[i,trial+1,modality] += (1 - np.exp(-iti/tauContext)) * (0.5 - pContext[i,trial+1,modality])
                    pContext[i,trial+1,(1 if modality==0 else 0)] = 1 - pContext[i,trial+1,modality]

                if not np.isnan(tauReinforcement):
                    qReinforcement[i,trial+1] *= np.exp(-iti/tauReinforcement)

                if not np.isnan(tauPerseveration):
                    qPerseveration[i,trial+1] *= np.exp(-iti/tauPerseveration)

                if not np.isnan(alphaResponse):
                    qResponse[i,trial+1] += (action[i,trial] - qResponse[i,trial]) * alphaResponse

                if not np.isnan(tauResponse):
                    qResponse[i,trial+1] *= np.exp(-iti/tauResponse)

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
        if any([w in prm for w in ('wContext','wReinforcement','wPerseveration','wResponse','wReward')]) and val > 0:
            p *= scipy.stats.norm(0,10).pdf(val)
    return p


def evalModel(params,*args):
    sessionData,trainTrials,trainingPhase,fixedInd,fixedVal,paramNames,paramsDict = args
    if fixedInd is not None:
        params = insertFixedParamVals(params,fixedInd,fixedVal)
    response = sessionData.trialResponse[trainTrials]
    prediction = runModel(sessionData,*params,**paramsDict)[-2][0][trainTrials]
    logLoss = sklearn.metrics.log_loss(response,prediction,normalize=False,sample_weight=None)
    logLoss += -np.log(calcPrior(params,paramNames))
    return logLoss


def fitModel(mouseId,sessionStartTime,trainingPhase,modelType,fixedParamsIndex):

    modelParams = {'visConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
                   'audConfidence': {'bounds': (0.5,1), 'fixedVal': 1},
                   'qInitVis': {'bounds': (0,1), 'fixedVal': 0},
                   'qInitAud': {'bounds': (0,1), 'fixedVal': 0},
                   'wContext': {'bounds': (0,30), 'fixedVal': 0},
                   'alphaContext': {'bounds':(0,1), 'fixedVal': np.nan},
                   'alphaContextNeg': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauContext': {'bounds': (1,360), 'fixedVal': np.nan},
                   'alphaContextReinforcement': {'bounds': (0,1), 'fixedVal': np.nan},
                   'wReinforcement': {'bounds': (0,30), 'fixedVal': 0},
                   'alphaReinforcement': {'bounds': (0,1), 'fixedVal': np.nan},
                   'alphaReinforcementNeg': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauReinforcement': {'bounds': (1,360), 'fixedVal': np.nan},
                   'wPerseveration': {'bounds': (0,30), 'fixedVal': 0},
                   'alphaPerseveration': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauPerseveration': {'bounds': (1,360), 'fixedVal': np.nan},
                   'wResponse': {'bounds': (0,30), 'fixedVal': 0},
                   'alphaResponse': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauResponse': {'bounds': (1,360), 'fixedVal': np.nan},
                   'wReward': {'bounds': (0,30), 'fixedVal': 0},
                   'alphaReward': {'bounds': (0,1), 'fixedVal': np.nan},
                   'tauReward': {'bounds': (1,60), 'fixedVal': np.nan},
                   'wBias': {'bounds':(0,30), 'fixedVal': 0}}

    fileName = str(mouseId)+'_'+sessionStartTime+'_'+trainingPhase+'_'+modelType+('' if fixedParamsIndex=='None' else '_'+fixedParamsIndex)+'.npz'
    filePath = os.path.join(baseDir,'Sam','RLmodel','learning',fileName)

    sessionData = getSessionData(mouseId,sessionStartTime)
    
    modelParamNames = list(modelParams.keys())
    paramsDict = {}

    # fitFunc = scipy.optimize.direct
    # fitFuncParams = {'eps': 1e-3,'maxfun': None,'maxiter': int(1e3),'locally_biased': False,'vol_tol': 1e-16,'len_tol': 1e-6}
    fitFunc = scipy.optimize.differential_evolution
    fitFuncParams = {'mutation': (0.5,1),'recombination': 0.7,'popsize': 20,'strategy': 'best1bin', 'init': 'sobol', 'workers': 1} 

    if modelType == 'BasicRL':
        coreFixedPrms = ['qInitVis','qInitAud','wContext','alphaContext','alphaContextNeg','tauContext','alphaContextReinforcement','alphaReinforcementNeg','tauReinforcement','wPerseveration','alphaPerseveration','tauPerseveration','wResponse','alphaResponse','tauResponse']
        fixedParams = [coreFixedPrms,
                       coreFixedPrms + ['visConfidence','audConfidence'],
                       coreFixedPrms + ['alphaReinforcement'],
                       coreFixedPrms + ['wReward','alphaReward','tauReward'],
                       [prm for prm in coreFixedPrms if prm not in ('alphaReinforcementNeg',)],
                       [prm for prm in coreFixedPrms if prm not in ('qInitVis','qInitAud','wContext','alphaContext','tauContext')]]
    elif modelType == 'ContextRL':
        coreFixedPrms = ['alphaContextNeg','alphaContextReinforcement','wReinforcement','alphaReinforcement','alphaReinforcementNeg','tauReinforcement','wPerseveration','alphaPerseveration','tauPerseveration','wResponse','alphaResponse','tauResponse']
        fixedParams = [coreFixedPrms,
                       coreFixedPrms + ['qInitVis','qInitAud'],
                       coreFixedPrms + ['tauContext'],
                       coreFixedPrms + ['wReward','alphaReward','tauReward'],
                       [prm for prm in coreFixedPrms if prm not in ('alphaContextNeg',)],
                       [prm for prm in coreFixedPrms if prm not in ('alphaContextReinforcement',)],
                       [prm for prm in coreFixedPrms if prm not in ('wReinforcement','alphaReinforcement')],
                       [prm for prm in coreFixedPrms if prm not in ('wReinforcement','alphaReinforcement','alphaReinforcementNeg')],
                       [prm for prm in coreFixedPrms if prm not in ('wPerseveration','alphaPerseveration','tauPerseveration')],
                       [prm for prm in coreFixedPrms if prm not in ('wResponse','alphaResponse','tauResponse')]]
    
    params = []
    logLossTrain = []
    logLossTest = []
    nIters = 5
    nFolds = 5
    nTrials = sessionData.nTrials
    n = round(nTrials / nFolds)
    for fixedPrms in (fixedParams if fixedParamsIndex=='None' else (fixedParams[int(fixedParamsIndex)],)):
        fixedParamIndices = [modelParamNames.index(prm) for prm in fixedPrms]
        fixedParamValues = [modelParams[prm]['fixedVal'] for prm in fixedPrms]
        bounds = tuple(modelParams[prm]['bounds'] for  prm in modelParamNames if prm not in fixedPrms)
        params.append([])
        logLossTrain.append([])
        logLossTest.append([])
        for _ in range(nIters):
            shuffleInd = np.random.permutation(nTrials)
            prediction = np.full(nTrials,np.nan)
            for k in range(nFolds):
                start = k * n
                testTrials = shuffleInd[start:start+n] if k+1 < nFolds else shuffleInd[start:]
                trainTrials = np.setdiff1d(shuffleInd,testTrials)
                fit = fitFunc(evalModel,bounds,args=(sessionData,trainTrials,trainingPhase,fixedParamIndices,fixedParamValues,modelParamNames,paramsDict),**fitFuncParams)
                params[-1].append(insertFixedParamVals(fit.x,fixedParamIndices,fixedParamValues))
                logLossTrain[-1].append(fit.fun)
                prediction[testTrials] = runModel(sessionData,*params[-1][-1],**paramsDict)[-2][0][testTrials]
            logLossTest[-1].append(sklearn.metrics.log_loss(sessionData.trialResponse,prediction,normalize=True))

    np.savez(filePath,params=params,logLossTrain=logLossTrain,logLossTest=logLossTest,**paramsDict) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=str)
    parser.add_argument('--sessionStartTime',type=str)
    parser.add_argument('--trainingPhase',type=str)
    parser.add_argument('--modelType',type=str)
    parser.add_argument('--fixedParamsIndex',type=str)
    args = parser.parse_args()
    trainingPhase = args.trainingPhase.replace('_',' ')
    fitModel(args.mouseId,args.sessionStartTime,trainingPhase,args.modelType,args.fixedParamsIndex)
