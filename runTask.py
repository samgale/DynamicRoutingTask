# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:48:22 2021

@author: svc_ccg
"""

import argparse
import json
import os
import shutil
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
    
localTaskDir = os.path.dirname(paramsPath)
#    
#localTaskPath = os.path.join(localTaskDir,os.path.basename(params['taskScript']))
#
#shutil.copy2(params['taskScript'],localTaskPath)
#
#os.chdir(localTaskDir)

localTaskPath = "C:\Users\svc_neuropix\Desktop\TaskControl.py"

localTaskPath = params['taskScript']

toRun = ('call activate ' + env + '\n' +
'python ' + '"' + localTaskPath + '" ' + '"' + paramsPath + '"')

print(toRun)

#errorPath = os.path.join(localTaskDir,'error.log')
#outputPath = os.path.join(localTaskDir,'output.log')
#
#print(outputPath)
#print(errorPath)
#
#f = open(outputPath,'wb')
#f.close()
#
#f = open(errorPath,'wb')
#f.close()
#
#with open(outputPath,'wb') as out, open(errorPath,'wb') as err:
#    p = subprocess.Popen(toRun,stdout=out,stderr=err)
    
p = subprocess.Popen(toRun)