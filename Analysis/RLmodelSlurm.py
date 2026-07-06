# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os
import numpy as np
from simple_slurm import Slurm
from RLmodelHPC import getSessions

# script to run
script_path = '/allen/ai/homedirs/samg/PythonScripts/RLmodelHPC.py'

# job record output folder
stdout_location = '/allen/ai/homedirs/samg/job_records'

# python path
baseDir ='/allen/programs/mindscope/workgroups/dynamicrouting'
python_path = os.path.join(baseDir,'Sam/miniconda/envs/RLmodel/bin/python')

# call the `sbatch` command to run the jobs
dirName = 'contextBelief'

# trainingPhases = ('initial training','early learning','late learning','after learning','sessionClusters',
#                   'opto','ephys','nogo','noAR','rewardOnly','no reward')

if dirName == 'noiseSim':
    cpus = 20
    mem = '2gb'
    modelTypes = ('ContextRL',)
    trainingPhases = ('after learning',)
    fixedParamsIndices = [0,1,2]
else:
    cpus = 1
    mem = '1gb'
    if dirName == 'learning':
        modelTypes = ('BasicRL','ContextRL')
        trainingPhases = ('initial training','early learning','late learning','after learning')
    elif dirName == 'contextBelief':
        modelTypes = ('ContextRL',)
        trainingPhases = ('after learning',)
    fixedParamsIndices = None # list of ints or None

slurm = Slurm(cpus_per_task=cpus,
              partition='braintv',
              job_name='RLmodel',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='144:00:00',
              mem_per_cpu=mem)

for trainingPhase in trainingPhases:
    if dirName == 'noiseSim':
        mice = ['all']
        sessions = [['all_all']]
    elif dirName == 'contextBelief':
        d = np.load(os.path.join(baseDir,'Sam','contextBelief.npy'),allow_pickle=True).item()
        mice = list(d.keys())
        sessions = [list(d[m].keys()) for m in mice]
    else:
        mice,sessions = getSessions(trainingPhase)
    for mouseId,startTimes in zip(mice,sessions):
        for sessionStartTime in startTimes:
            for modelType in modelTypes:
                for fixedParamsIndex in ((None,) if fixedParamsIndices is None else fixedParamsIndices):
                    slurm.sbatch('{} {} --dirName {} --mouseId {} --sessionStartTime {} --trainingPhase {} --modelType {} --fixedParamsIndex {}'.format(
                                 python_path,script_path,dirName,mouseId,sessionStartTime,trainingPhase.replace(' ','_'),modelType,fixedParamsIndex))
