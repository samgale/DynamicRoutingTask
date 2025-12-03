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
slurm = Slurm(cpus_per_task=1,
              partition='braintv',
              job_name='RLmodel',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='72:00:00',
              mem_per_cpu='1gb')

modelTypes = ('BasicRL','ContextRL')

trainingPhases = ('initial training','early learning','late learning','after learning','sessionClusters',
                  'opto','ephys','nogo','noAR','rewardOnly','no reward')

nFixedParamSets = None # int or None

for trainingPhase in ('initial training','early learning','late learning','after learning'):
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
            summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
            summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
            drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
            if trainingPhase in ('initial training','early learning','late learning','after learning','ephys'):
                hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])
                ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['cannula'] & ~summaryDf['stage 5 repeats']
                mice = np.array(summaryDf[ind]['mouse id'])
                sessions = []
                if trainingPhase == 'ephys':
                    nonStandardTrainingMice = (644864,644866,644867,681532,686176)
                    mice = np.concatenate((mice,nonStandardTrainingMice))
                    for mouseId in mice:
                        df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
                        sessions.append(df['start time'][df['ephys'] & ~np.array(df['ignore']).astype(bool)])
                else:
                    for mouseId in mice:
                        df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
                        preExperimentSessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore']).astype(bool)
                        firstExperimentSession = getFirstExperimentSession(df)
                        if firstExperimentSession is not None:
                            preExperimentSessions[firstExperimentSession:] = False
                        preExperimentSessions = np.where(preExperimentSessions)[0]
                        sessionsToPass = getSessionsToPass(mouseId,df,preExperimentSessions,stage=5)
                        learnOnset = np.load(os.path.join(baseDir,'Sam','learnOnset.npy'),allow_pickle=True).item()[mouseId]
                        nSessionsToFit = 2
                        if trainingPhase == 'initial training':
                            sessions.append(df.loc[preExperimentSessions,'start time'][:nSessionsToFit])
                        elif trainingPhase == 'early learning':
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
    for mouseId,startTimes in zip(mice,sessions):
        for sessionStartTime in startTimes:
            for modelType in modelTypes:
                for fixedParamsIndex in ((None,) if nFixedParamSets is None else range(nFixedParamSets)):
                    slurm.sbatch('{} {} --mouseId {} --sessionStartTime {} --trainingPhase {} --modelType {} --fixedParamsIndex {}'.format(
                                 python_path,script_path,mouseId,sessionStartTime,trainingPhase.replace(' ','_'),modelType,fixedParamsIndex))
