# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:36:05 2021

@author: svc_ccg
"""

import os
import argparse
from zro import Proxy

computerName = {'NP2': 'w10DT713937',
                'NP3': 'w10DT713941',
                'B1': 'wxvs-syslogic7',
                'B2': 'wxvs-syslogic8',
                'B3': 'wxvs-syslogic9',
                'B4': 'wxvs-syslogic10',
                'B5': 'wxvs-syslogic11',
                'B6': 'wxvs-syslogic12',
                'E1': 'w10dtmj0jcmzd',
                'E2': 'w10dtmj0jcmza',
                'E3': 'w10dtmj0jcmz4',
                'E4': 'w10dtmj0jcmzc',
                'E5': 'w10dtmj0jcmzf',
                'E6': 'w10dtmj0jcmyz',
                'F1': 'w10dtmj0jcmz0',
                'F2': 'w10dtmj0jcmz2',
                'F3': 'w10dtmj0jcmz5',
                'F4': 'w10dtmj0jcmz6',
                'F5': 'w10dtmj0jcmz7',
                'F6': 'w10dtmj0jcmzb'}

runTask = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\runTask.py"

paramNames = ('rigName','subjectName','taskScript','taskVersion','rewardSound','saveSoundArray','optoParamsPath','optoTaggingLocs',
              'galvoX','galvoY','galvoDwellTime','optoDev','optoAmp','optoDur','optoFreq','optoOffset')

parser = argparse.ArgumentParser()
for prm in paramNames:
    parser.add_argument('--'+prm)

args = parser.parse_args()

paramsDict = {prm: getattr(args,prm) for prm in paramNames}

if paramsDict['taskScript'][:4] == 'http':
    urlDir = os.path.dirname(paramsDict['taskScript'])
    paramsDict['task_script_commit_hash'] = os.path.basename(urlDir)
    paramsDict['GHTaskScriptParams'] = {'taskScript': paramsDict['taskScript'],
                                        'taskControl': urlDir+'/TaskControl.py',
                                        'optoParams': urlDir+'/OptoParams.py'}
    import requests
    response = requests.get(urlDir+'/runTask.py')
    if not response.status_code in (200, ):
        response.raise_for_status()
    runTask = response.content

agent = Proxy(computerName[args.rigName] + ':5000')
agent.start_script(script=runTask, params=paramsDict)
