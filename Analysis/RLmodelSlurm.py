# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os
import numpy as np
import pandas as pd
from simple_slurm import Slurm

# script to run
script_path = '/allen/ai/homedirs/samg/PythonScripts/RLmodelHPC.py'

# job record output folder
stdout_location = '/allen/ai/homedirs/samg/job_records'

# python path
python_path = '/allen/programs/mindscope/workgroups/dynamicrouting/Sam/miniconda/envs/RLmodel/bin/python'

# call the `sbatch` command to run the jobs
slurm = Slurm(cpus_per_task=1,
              partition='braintv',
              job_name='RLmodel',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='24:00:00',
              mem_per_cpu='1gb')

summarySheets = pd.read_excel('/allen/programs/mindscope/workgroups/dynamicrouting/Sam/BehaviorSummary.xlsx',sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
trainingPhases = ('initial training','after learning','nogo','noAR','rewardOnly','no reward')
for trainingPhase in trainingPhases:
    if trainingPhase in ('initial training','after learning'):
        hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])
        ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['cannula'] & ~summaryDf['stage 5 repeats']
        mice = np.array(summaryDf[ind]['mouse id'])
        nSessions = [5] * len(mice)
    else:
        mice = np.array(summaryDf[summaryDf[trainingPhase]]['mouse id'])
        drSheets,nsbSheets = [pd.read_excel(os.path.join('/allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
        nSessions = []
        for mouseId in mice:
            df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
            sessions = np.array([trainingPhase in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
            nSessions.append(sessions.sum()) 
    for mouseId,n in zip(mice,nSessions):
        for sessionIndex in range(n):
            slurm.sbatch('{} {} --mouseId {} --sessionIndex {} --trainingPhase {}'.format(
                         python_path,script_path,mouseId,sessionIndex,trainingPhase.replace(' ','_')))
