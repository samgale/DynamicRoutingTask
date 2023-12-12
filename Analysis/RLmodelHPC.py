# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import argparse
import itertools
import os
import h5py
import numpy as np
import pandas as pd
import sklearn
from  DynamicRoutingAnalysisUtils import getFirstExperimentSession,getSessionsToPass,getSessionData
from RLmodelUtils import runModel


baseDir = '//allen/programs/mindscope/workgroups/dynamicrouting'
drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)


def dictToHdf5(group,d):
    for key,val in d.items():
        if isinstance(val,dict): 
            dictToHdf5(group.create_group(key),val)
        else:
            group.create_dataset(key,data=val)


def fitModel(mouseId,trainingPhase,contextMode,qMode):
    nSessions = 5
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
        
    visConfidenceRange = np.arange(0.75,1,0.1)
    audConfidenceRange = np.arange(0.65,1,0.1)
    if contextMode == 'no context':
        alphaContextRange = (0,)
    else:
        alphaContextRange = np.arange(0.05,1,0.15) 
    tauActionRange = (0.25,)
    biasActionRange = np.arange(0.05,1,0.15)
    if qMode == 'no q update':
        alphaActionRange = (0,)
    else:
        alphaActionRange = np.arange(0.02,0.11,0.02) if trainingPhase=='initial training' else np.arange(0.05,1,0.15)
    penaltyRange = (-1,)
    fitParamRanges = (visConfidenceRange,audConfidenceRange,
                      alphaContextRange,
                      tauActionRange,biasActionRange,alphaActionRange,
                      penaltyRange)
    
    modelParams = []
    modelResponse = []
    for testExp in exps:
        trainExps = [obj for obj in exps if obj is not testExp]
        actualResponse = np.concatenate([obj.trialResponse for obj in trainExps])
        minLoss = 1000
        for params in itertools.product(*fitParamRanges):
            trainResponse = np.concatenate([np.mean(runModel(obj,contextMode,*params)[0],axis=0) for obj in trainExps])
            logLoss = sklearn.metrics.log_loss(actualResponse,trainResponse)
            if logLoss < minLoss:
                minLoss = logLoss
                fitParams = params
        modelParams.append(fitParams)
        modelResponse.append(np.mean(runModel(testExp,contextMode,*fitParams)[0],axis=0))
    
    fileName = str(mouseId)+'_'+trainingPhase+'_'+contextMode+'_'+qMode+'.hdf5'
    h5File = h5py.File(os.path.join(baseDir,'Sam','RLmodel',fileName),'w')
    d = {trainingPhase: {contextMode: {qMode: {'modelParams': modelParams, 'modelResponse': modelResponse}}}}
    dictToHdf5(h5File,d)
    h5File.close()    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouseId',type=int)
    parser.add_argument('--trainingPhase',type=str)
    parser.add_argument('--contextMode',type=str)
    parser.add_argument('--qMode',type=str)
    args = parser.parse_args()
    fitModel(args.mouseId,args.trainingPhase,args.contextMode,args.qMode)
