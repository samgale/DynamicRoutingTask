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
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getSessionsToPass,getFirstExperimentSession,getSessionData


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting')


def getSessions(trainingPhase):
    if trainingPhase == 'sessionClusters':
        sessionClustData = np.load(os.path.join(baseDir,'Sam','sessionClustData.npy'),allow_pickle=True).item()
        clustersToFit = (4,6)
        nSessionsToFit = 4
        mice = []
        sessions = []
        for mouseId in np.unique(sessionClustData['mouseId']):
            isMouse = sessionClustData['mouseId']==mouseId
            if all([np.sum(isMouse & (sessionClustData['clustId']==clust)) >= nSessionsToFit for clust in clustersToFit]):
                mice.append(mouseId)
                sessions.append([])
                for clust in clustersToFit:
                    sessions[-1].extend(sessionClustData['sessionStartTime'][isMouse & (sessionClustData['clustId']==clust)][:nSessionsToFit])
    else:
        if trainingPhase == 'opto':
            optoLabel = 'lFC'
            optoExps = pd.read_excel(os.path.join(baseDir,'Sam','OptoExperiments.xlsx'),sheet_name=None)
            mice = []
            sessions = []
            for mouseId in optoExps:
                df = optoExps[mouseId]
                sessions = df[optoLabel] & ~(df['unilateral'] & df['bilateral'])
                if any(sessions):
                    mice.append(mouseId)
                    sessions.append(df['start time'])
        else:
            summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
            summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
            drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
            if trainingPhase == 'ephys':
                sessionTable = pd.read_parquet('/allen/programs/mindscope/workgroups/dynamicrouting/Ethan/CO decoding results/session_table_v0.272.parquet')
                sessionTable = sessionTable[sessionTable.is_production & sessionTable.is_annotated & ~sessionTable.is_templeton]
                mice = np.unique(sessionTable.subject_id)
                sessions = []
                for mouseId in mice:
                    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
                    sessions.append([dt for dt,st in zip(df['start time'],df['start time'].astype(str)) if st[:10] in list(sessionTable[sessionTable.subject_id==mouseId].date)])
            elif trainingPhase in ('initial training','early learning','late learning','after learning',):
                isStandardRegimen = getIsStandardRegimen(summaryDf)
                mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])
                sessions = []
                for mouseId in mice:
                    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
                    preExperimentSessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore']).astype(bool)
                    firstExperimentSession = getFirstExperimentSession(df)
                    if firstExperimentSession is not None:
                        preExperimentSessions[firstExperimentSession:] = False
                    preExperimentSessions = np.where(preExperimentSessions)[0]
                    sessionsToPass = getSessionsToPass(mouseId,df,preExperimentSessions,stage=5)
                    nSessionsToFit = 2
                    if trainingPhase == 'initial training':
                        sessions.append(df.loc[preExperimentSessions,'start time'][:nSessionsToFit])
                    elif trainingPhase == 'early learning':
                        learnOnset = np.load(os.path.join(baseDir,'Sam','learnOnset.npy'),allow_pickle=True).item()[mouseId]
                        sessions.append(df.loc[preExperimentSessions,'start time'][learnOnset+1:learnOnset+1+nSessionsToFit])
                    elif trainingPhase == 'late learning':
                        sessions.append(df.loc[preExperimentSessions,'start time'][sessionsToPass-2-nSessionsToFit:sessionsToPass-2])
                    elif trainingPhase == 'after learning':
                        sessions.append(df.loc[preExperimentSessions,'start time'][sessionsToPass:sessionsToPass+nSessionsToFit])
            else:
                mice = np.array(summaryDf[summaryDf[trainingPhase]]['mouse id'])
                sessions = []
                for mouseId in mice:
                    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
                    sessions.append(df['start time'][np.array([trainingPhase in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))])
        sessions = [[st.strftime('%Y%m%d_%H%M%S') for st in startTimes] for startTimes in sessions]
    return mice,sessions


def getMeanBlockSwitchResponse(sessionData,modelResp=None):
    preTrials = 5
    resp = []
    for rewardStim in ('vis1','sound1'):
        for stim in ('vis1','sound1','vis2','sound2'):
            y = []
            for m,exps in enumerate(sessionData):
                y.append([])
                for s,obj in enumerate(exps):
                    trials = obj.trialStim==stim
                    r = obj.trialResponse if modelResp is None else modelResp[m][s]
                    for blockInd,blockRewStim in enumerate(obj.blockStimRewarded):
                        if blockInd > 0 and blockRewStim==rewardStim:
                            postTrials = 20 if stim==blockRewStim else 15
                            y[-1].append(np.full(preTrials+postTrials,np.nan))
                            pre = r[(obj.trialBlock==blockInd) & trials]
                            i = min(preTrials,pre.size)
                            y[-1][-1][preTrials-i:preTrials] = pre[-i:]
                            post = r[(obj.trialBlock==blockInd+1) & trials]
                            i = min(postTrials,post.size)
                            y[-1][-1][preTrials:preTrials+i] = post[:i]
                y[-1] = np.nanmean(y[-1],axis=0)
            resp.append(np.nanmean(y,axis=0))
    return np.concatenate(resp)


def runModel(obj,visConfidence,audConfidence,qInitVis,qInitAud,
             wContext,alphaContext,alphaContextNeg,tauContext,alphaContextReinforcement,
             wReinforcement,alphaReinforcement,alphaReinforcementNeg,tauReinforcement,
             wPerseveration,alphaPerseveration,tauPerseveration,wResponse,alphaResponse,tauResponse,
             wReward,alphaReward,tauReward,wBias,
             sigmaContext=0,sigmaBias=0,contextBelief=None,noAgent=[],useChoiceHistory=True,nReps=1):

    stimNames = ('vis1','vis2','sound1','sound2')
    stimConfidence = [visConfidence,audConfidence]
    modality = 0

    pContext = 0.5 + np.zeros((nReps,obj.nTrials,2))
    if contextBelief is not None and np.isnan(alphaContext):
        pContext[:,:,0] = contextBelief
        pContext[:,:,1] = 1 - contextBelief

    qContext = np.zeros((nReps,obj.nTrials,2,len(stimNames)))
    qContext[:,0,0] = [1,0,qInitAud,0]
    qContext[:,0,1] = [qInitVis,0,1,0]
    
    qReinforcement = np.zeros((nReps,obj.nTrials,len(stimNames)))
    qReinforcement[:,0] = [1,0,1,0]

    qPerseveration = np.zeros((nReps,obj.nTrials,len(stimNames)))

    qResponse = np.zeros((nReps,obj.nTrials))

    qReward = np.zeros((nReps,obj.nTrials))
    
    bias = wBias

    qTotal = np.zeros((nReps,obj.nTrials))

    pAction = np.zeros((nReps,obj.nTrials))
    
    action = np.zeros((nReps,obj.nTrials),dtype=int)
    
    for i in range(nReps):
        for trial,stim in enumerate(obj.trialStim):
            if stim != 'catch':
                modality = 0 if 'vis' in stim else 1
                pStim = np.zeros(len(stimNames))
                pStim[[stim[:-1] in s for s in stimNames]] = [stimConfidence[modality],1-stimConfidence[modality]] if '1' in stim else [1-stimConfidence[modality],stimConfidence[modality]]

                pState = pStim[None,:] * pContext[i,trial][:,None]

                expectedOutcomeContext = 0 if 'context' in noAgent else np.sum(pState * qContext[i,trial])

                expectedOutcomeReinforcement = 0 if 'reinforcement' in noAgent else np.sum(pStim * qReinforcement[i,trial])

                expectedAction = 0 if 'perseveration' in noAgent else np.sum(pStim * qPerseveration[i,trial])

                qResp = 0 if 'response' in noAgent else qResponse[i,trial]

                qRew = 0 if 'reward' in noAgent else qReward[i,trial]

                qTotal[i,trial] = ((wContext * (2*expectedOutcomeContext-1)) + 
                                   (wReinforcement * (2*expectedOutcomeReinforcement-1)) + 
                                   (wPerseveration * (2*expectedAction-1)) + 
                                   (wResponse * (2*qResp-1)) + 
                                   (wReward * (2*qRew-1)) + 
                                   bias)

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
                            outcomeError = pState * (reward - qContext[i,trial])
                            qContext[i,trial+1] += outcomeError * alphaContextReinforcement
                        
                        if not np.isnan(alphaReinforcement):
                            outcomeError = pStim * (reward - qReinforcement[i,trial])
                            qReinforcement[i,trial+1] += outcomeError * (alphaReinforcementNeg if not np.isnan(alphaReinforcementNeg) and not reward else alphaReinforcement)
                    
                    if not np.isnan(alphaPerseveration):
                        actionError = pStim * (action[i,trial] - qPerseveration[i,trial])
                        qPerseveration[i,trial+1] += actionError * alphaPerseveration
                
                iti = obj.stimStartTimes[trial+1] - obj.stimStartTimes[trial]

                if not np.isnan(alphaContext):
                    if not np.isnan(tauContext):
                        pContext[i,trial+1,modality] += (1 - np.exp(-iti/tauContext)) * (0.5 - pContext[i,trial+1,modality])
                    if sigmaContext > 0:
                        pContext[i,trial+1,modality] += random.gauss(0,sigmaContext)
                        pContext[i,trial+1,modality] = np.clip(pContext[i,trial+1,modality],0,1)
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
                    
                if sigmaBias > 0:
                    bias += wBias * random.gauss(0,sigmaBias)
    
    return pContext, qReinforcement, qPerseveration, qResponse, qReward, qTotal, pAction, action


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


def calcL2Error(params,paramNames):
    sigma = 10
    r = 0
    for prm,val in zip(paramNames,params):
        if any([w in prm for w in ('wContext','wReinforcement','wPerseveration','wResponse','wReward')]) and val > 0:
            r += 1 / (2 * sigma**2) * val**2 
    return r


def evalModel(params,*args):
    sessionData,trainTrials,trainingPhase,fixedInd,fixedVal,paramNames,paramsDict = args
    if fixedInd is not None:
        params = insertFixedParamVals(params,fixedInd,fixedVal)
    if trainingPhase == 'noiseSim':
        response = getMeanBlockSwitchResponse(sessionData)
        modelResp = [[np.mean(runModel(obj,*params,**paramsDict,useChoiceHistory=False,nReps=5)[-2],axis=0) for obj in s] for s in sessionData]
        prediction = getMeanBlockSwitchResponse(sessionData,modelResp)
        mse = np.mean((response - prediction)**2)
        mse += calcL2Error(params,paramNames)
        return mse
    else:
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
                   'wBias': {'bounds': (0,30), 'fixedVal': 0},
                   'sigmaContext': {'bounds': (0,0.25), 'fixedVal': 0}}

    dirName = 'contextBelief'
    fileName = str(mouseId)+'_'+sessionStartTime+'_'+trainingPhase+'_'+modelType+('' if fixedParamsIndex=='None' else '_'+fixedParamsIndex)+'.npz'
    filePath = os.path.join(baseDir,'Sam','RLmodel',dirName,fileName)
    
    modelParamNames = list(modelParams.keys())

    # fitFunc = scipy.optimize.direct
    # fitFuncParams = {'eps': 1e-3,'maxfun': None,'maxiter': int(1e3),'locally_biased': False,'vol_tol': 1e-16,'len_tol': 1e-6}
    fitFunc = scipy.optimize.differential_evolution
    fitFuncParams = {'mutation': (0.5,1),'recombination': 0.7,'popsize': 20,'strategy': 'best1bin', 'init': 'sobol', 'workers': (20 if trainingPhase=='noiseSim' else 1)} 

    if modelType == 'BasicRL':
        coreFixedPrms = ['qInitVis','qInitAud','wContext','alphaContext','alphaContextNeg','tauContext','alphaContextReinforcement','alphaReinforcementNeg','tauReinforcement','wPerseveration','alphaPerseveration','tauPerseveration','wResponse','alphaResponse','tauResponse','sigmaContext']
        fixedParams = [coreFixedPrms,
                       coreFixedPrms + ['visConfidence','audConfidence'],
                       coreFixedPrms + ['wReward','alphaReward','tauReward'],
                       coreFixedPrms + ['alphaReinforcement'],
                       [prm for prm in coreFixedPrms if prm not in ('alphaReinforcementNeg',)],
                       [prm for prm in coreFixedPrms if prm not in ('wContext','alphaContext','tauContext')]]
    elif modelType == 'ContextRL':
        coreFixedPrms = ['qInitVis','qInitAud','alphaContextNeg','alphaContextReinforcement','wReinforcement','alphaReinforcement','alphaReinforcementNeg','tauReinforcement','wPerseveration','alphaPerseveration','tauPerseveration','wResponse','alphaResponse','tauResponse','sigmaContext']
        if trainingPhase == 'noiseSim':
            fixedParams = [coreFixedPrms,
                           coreFixedPrms + ['tauContext'],
                           [prm for prm in coreFixedPrms + ['tauContext'] if prm not in ('sigmaContext',)]]
        elif trainingPhase == 'contextBelief':
            fixedParams = [coreFixedPrms,
                           coreFixedPrms + ['tauContext'],
                           coreFixedPrms + ['alphaContext','tauContext']]
        else:
            fixedParams = [coreFixedPrms,
                           coreFixedPrms + ['visConfidence','audConfidence'],
                           coreFixedPrms + ['wReward','alphaReward','tauReward'],
                           coreFixedPrms + ['tauContext'],
                           [prm for prm in coreFixedPrms if prm not in ('alphaContextNeg',)],
                           [prm for prm in coreFixedPrms if prm not in ('alphaContextReinforcement')],
                           [prm for prm in coreFixedPrms if prm not in ('wReinforcement','alphaReinforcement')],
                           [prm for prm in coreFixedPrms + ['tauContext'] if prm not in ('wReinforcement','alphaReinforcement')],
                           [prm for prm in coreFixedPrms if prm not in ('wPerseveration','alphaPerseveration','tauPerseveration')],
                           [prm for prm in coreFixedPrms if prm not in ('wResponse','alphaResponse','tauResponse')]]
    
    if trainingPhase == 'noiseSim':
        mice,sessions = getSessions('after learning')
        sessionData = [[getSessionData(m,st,lightLoad=True) for st in s] for m,s in zip(mice,sessions)]
    else:
        sessionData = getSessionData(mouseId,sessionStartTime,lightLoad=True)
    
    paramsDict = {}
    if trainingPhase == 'contextBelief':
        contextBelief = np.load(os.path.join(baseDir,'Sam','contextBelief.npy'),allow_pickle=True).item()
        paramsDict['contextBelief'] = contextBelief[mouseId][sessionStartTime]
                   
    params = []
    logLossTrain = []
    logLossTest = []
    for fixedPrms in (fixedParams if fixedParamsIndex=='None' else (fixedParams[int(fixedParamsIndex)],)):
        fixedParamIndices = [modelParamNames.index(prm) for prm in fixedPrms]
        fixedParamValues = [modelParams[prm]['fixedVal'] for prm in fixedPrms]
        bounds = tuple(modelParams[prm]['bounds'] for  prm in modelParamNames if prm not in fixedPrms)
        params.append([])
        logLossTrain.append([])
        logLossTest.append([])
        if trainingPhase == 'noiseSim':
            trainTrials = None
            fit = fitFunc(evalModel,bounds,args=(sessionData,trainTrials,trainingPhase,fixedParamIndices,fixedParamValues,modelParamNames,paramsDict),**fitFuncParams)
            params[-1].append(insertFixedParamVals(fit.x,fixedParamIndices,fixedParamValues))
        else:
            nIters = 5
            nFolds = 5
            nTrials = sessionData.nTrials
            n = round(nTrials / nFolds)
            for _ in range(nIters):
                shuffleInd = np.random.permutation(nTrials)
                prediction = np.full(nTrials,np.nan)
                for k in range(nFolds):
                    start = k * n
                    testTrials = shuffleInd[start:start+n] if k+1 < nFolds else shuffleInd[start:]
                    trainTrials = np.setdiff1d(shuffleInd,testTrials)
                    fit = fitFunc(evalModel,bounds,args=(sessionData,trainTrials,trainingPhase,fixedParamIndices,fixedParamValues,modelParamNames,paramsDict),**fitFuncParams)
                    params[-1].append(insertFixedParamVals(fit.x,fixedParamIndices,fixedParamValues))
                    logLossTrain[-1].append(sklearn.metrics.log_loss(sessionData.trialResponse[trainTrials],runModel(sessionData,*params[-1][-1],**paramsDict)[-2][0][trainTrials],normalize=True))
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
