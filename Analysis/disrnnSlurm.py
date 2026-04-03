# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os
from simple_slurm import Slurm

# script to run
script_path = '/allen/ai/homedirs/samg/PythonScripts/disrnnHPC.py'

# job record output folder
stdout_location = '/allen/ai/homedirs/samg/job_records'

# python path
baseDir ='/allen/programs/mindscope/workgroups/dynamicrouting'
python_path = os.path.join(baseDir,'Sam/miniconda/envs/DisRNN/bin/python')

# call the `sbatch` command to run the jobs
nProcesses = 64
slurm = Slurm(cpus_per_task=nProcesses,
              partition='braintv',
              job_name='disrnn',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='48:00:00',
              mem_per_cpu='4gb')#,
              #gres='gpu:1 --constraint="a100|v100|l40s"')

for trainingPhase in ('initial_training','after_learning'):
    for rep in range(3):
        slurm.sbatch('{} {} --rep {} --nProcesses {} --trainingPhase {}'.format(python_path,script_path,rep,nProcesses,trainingPhase))
