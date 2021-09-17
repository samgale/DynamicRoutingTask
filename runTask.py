# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:48:22 2021

@author: svc_ccg
"""

import argparse
import json
import os
import subprocess

env = 'DynamicRoutingTaskDev'

parser = argparse.ArgumentParser()
parser.add_argument('params', type=str,
                    help='path to params file')
parser.add_argument('-o', type=str,
                    help='path to output file')

args = parser.parse_args()

paramsPath = args.params

with open(paramsPath,'r') as f:
    params = json.load(f)
    
toRun = ('call activate ' + env + '\n' +
'python ' + '"' + params['taskScript'] + '" ' + '"' + paramsPath + '"')

batFile = os.path.join(os.path.dirname(paramsPath),'toRun.bat')

with open(batFile,'w') as f:
    f.write(toRun)
    
p = subprocess.Popen([batFile])
p.wait()