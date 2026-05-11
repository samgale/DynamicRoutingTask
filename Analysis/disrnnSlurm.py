# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os
import numpy as np
import pandas as pd
from simple_slurm import Slurm
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getRNNSessions

# script to run
script_path = '/allen/ai/homedirs/samg/PythonScripts/disrnnHPC.py'

# job record output folder
stdout_location = '/allen/ai/homedirs/samg/job_records'

# python path
baseDir ='/allen/programs/mindscope/workgroups/dynamicrouting'
python_path = os.path.join(baseDir,'Sam/miniconda/envs/DisRNN/bin/python')

# call the `sbatch` command to run the jobs
usePooledData = True
if usePooledData:
    nProcesses = 24
    nReps = 3 if nProcesses > 1 else 64
    trainingPhases = ('initial_training','after_learning','noAR')
    mouseIds = ('pooled',)
    maxTrainSessions = 0
else:
    nProcesses = 25
    nReps = 3
    trainingPhases = ('after learning',)
    maxTrainSessions = 20
    summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
    summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
    drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
    isStandardRegimen = getIsStandardRegimen(summaryDf)
    mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])
    mouseIds = []
    for mouseId in mice:
        df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
        sessions = getRNNSessions(mouseId,df)
        if len(sessions) > maxTrainSessions:
            mouseIds.append(mouseId)
            
slurm = Slurm(cpus_per_task=nProcesses,
              partition='braintv',
              job_name='disrnn',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='48:00:00',
              mem_per_cpu='4gb',)
              #gres='gpu:1')

for mouseId in mouseIds:
    for trainingPhase in trainingPhases:
        for rep in range(nReps):
            slurm.sbatch('{} {} --rep {} --nProcesses {} --trainingPhase {} --mouseId {} --maxTrainSessions {}'.format(python_path,script_path,rep,nProcesses,trainingPhase,mouseId,maxTrainSessions))
