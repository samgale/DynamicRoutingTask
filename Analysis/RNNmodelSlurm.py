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
script_path = '/allen/ai/homedirs/samg/PythonScripts/RNNmodelHPC.py'

# job record output folder
stdout_location = '/allen/ai/homedirs/samg/job_records'

# python path
baseDir ='/allen/programs/mindscope/workgroups/dynamicrouting'
python_path = os.path.join(baseDir,'Sam/miniconda/envs/RNNmodel/bin/python')

# call the `sbatch` command to run the jobs
nProcesses = 50
slurm = Slurm(cpus_per_task=nProcesses,
              partition='braintv',
              job_name='RNNmodel',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='48:00:00',
              mem_per_cpu='2gb',
              gres='gpu:1 --constraint="a100|v100|l40s"')

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
isStandardRegimen = getIsStandardRegimen(summaryDf)
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])

maxTrainSessions = 20
mouseIds = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = getRNNSessions(mouseId,df)
    if len(sessions) > maxTrainSessions:
        mouseIds.append(mouseId)

for mouseId in mouseIds[6:]:
    print(mouseId)
    slurm.sbatch('{} {} --mouseId {} --maxTrainSessions {} --nProcesses {}'.format(
                 python_path,script_path,mouseId,maxTrainSessions,nProcesses))
