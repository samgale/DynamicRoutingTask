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
              mem_per_cpu='32gb')

summarySheets = pd.read_excel('//allen/programs/mindscope/workgroups/dynamicrouting/Sam/BehaviorSummary.xlsx',sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])
ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['cannula'] & ~summaryDf['stage 5 repeats']
mice = np.array(summaryDf[ind]['mouse id'])
trainingPhases = ('initial training','after learning')
contextModes = ('no context','switch context','weight context')
qModes = ('q update','no q update')
for mouseId in mice:
    for trainingPhase in trainingPhases:
        for contextMode in contextModes:
            for qMode in qModes:
                if (trainingPhase=='initial training' and contextMode!='no context') or (contextMode=='no context' and qMode=='no q update'):
                    continue
                slurm.sbatch('{} {} --sessionId {} --trainingPhase {} --contextMode {} --qMode {}'.format(python_path,script_path,mouseId,trainingPhase,contextMode,qMode))
