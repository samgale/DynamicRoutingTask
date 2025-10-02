# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os
import numpy as np
import pandas as pd
from simple_slurm import Slurm
from  DynamicRoutingAnalysisUtils import getFirstExperimentSession, getSessionsToPass

# script to run
script_path = '/allen/ai/homedirs/samg/PythonScripts/RLmodelHPC.py'

# job record output folder
stdout_location = '/allen/ai/homedirs/samg/job_records'

# python path'
baseDir ='/allen/programs/mindscope/workgroups/dynamicrouting'
python_path = os.path.join(baseDir,'Sam/miniconda/envs/RLmodel/bin/python')

# call the `sbatch` command to run the jobs
slurm = Slurm(cpus_per_task=10,
              partition='braintv',
              job_name='RLmodel',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='72:00:00',
              mem_per_cpu='1gb')

modelTypes = ('ContextRL',)

trainingPhases = ('initial training','early learning','late learning','after learning','learning weights','opto','ephys',
                  'nogo','noAR','rewardOnly','no reward','clusters','cluster weights')

nSessionsToFit = 2
nFixedParamSets = None # int or None

for trainingPhase in trainingPhases[:4]:
    if trainingPhase == 'opto':
        optoLabel = 'lFC'
        optoExps = pd.read_excel(os.path.join(baseDir,'Sam','OptoExperiments.xlsx'),sheet_name=None)
        mice = []
        nSessions = []
        for mouseId in optoExps:
            df = optoExps[mouseId]
            sessions = df[optoLabel] & ~(df['unilateral'] & df['bilateral'])
            if any(sessions):
                mice.append(mouseId)
                nSessions.append(sum(sessions)) 
    else:
        summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
        summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
        drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
        if trainingPhase in ('initial training','early learning','late learning','after learning','learning weights','clusters','cluster weights','ephys'):
            hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])
            ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['cannula'] & ~summaryDf['stage 5 repeats']
            mice = np.array(summaryDf[ind]['mouse id'])
            if trainingPhase in ('clusters','cluster weights'):
                nSessions = []
                for mouseId in mice:
                    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
                    preExperimentSessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
                    firstExperimentSession = getFirstExperimentSession(df)
                    if firstExperimentSession is not None:
                        preExperimentSessions[firstExperimentSession:] = False
                    preExperimentSessions = np.where(preExperimentSessions)[0]
                    sessionsToPass = getSessionsToPass(mouseId,df,preExperimentSessions,stage=5)
                    sessions = preExperimentSessions[sessionsToPass:sessionsToPass+10]
                    nSessions.append(len(sessions))
            elif trainingPhase == 'ephys':
                nonStandardTrainingMice = (644864,644866,644867,681532,686176)
                mice = np.concatenate((mice,nonStandardTrainingMice))
                nSessions = []
                for mouseId in mice:
                    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
                    nSessions.append(df['ephys'].sum())
                nSessions = np.array(nSessions)
                hasEphys = nSessions > 0
                mice = mice[hasEphys]
                nSessions = nSessions[hasEphys]
            elif trainingPhase == 'learning weights':
                nSessions = [1] * len(mice)
            else:
                nSessions = [nSessionsToFit] * len(mice)
        else:
            mice = np.array(summaryDf[summaryDf[trainingPhase]]['mouse id'])
            nSessions = []
            for mouseId in mice:
                df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
                sessions = np.array([trainingPhase in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
                nSessions.append(sessions.sum()) 
    for mouseId,n in zip(mice,nSessions):
        for sessionIndex in range(n):
            for modelType in modelTypes:
                for fixedParamsIndex in ((None,) if nFixedParamSets is None else range(nFixedParamSets)):
                    slurm.sbatch('{} {} --mouseId {} --sessionIndex {} --trainingPhase {} --modelType {} --fixedParamsIndex {}'.format(
                                 python_path,script_path,mouseId,sessionIndex,trainingPhase.replace(' ','_'),modelType,fixedParamsIndex))
