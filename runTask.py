# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:48:22 2021

@author: svc_ccg
"""

import sys
import json
import subprocess

env = 'DynamicRoutingTaskDev'

paramsPath = sys.argv[2]
with open(paramsPath,'r') as f:
    params = json.load(f)
    
taskScript = params['taskScript']

toRun = """
call activate env
python {taskScript} {paramsPath}
"""

p = subprocess.Popen(toRun)