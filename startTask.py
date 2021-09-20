# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:36:05 2021

@author: svc_ccg
"""

import argparse
from zro import Proxy

computerName = {'NP3': 'w10DTSM118296',
                'E1': 'wxvs-syslogic31',
                'E2': 'wxvs-syslogic32',
                'E3': 'wxvs-syslogic33',
                'E4': 'wxvs-syslogic34',
                'E5': 'wxvs-syslogic35',
                'E6': 'wxvs-syslogic36'}

runTaskPath = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\DynamicRoutingTask\runTask.py'

paramNames = ('rigName','subjectName','taskScript','taskVersion')

parser = argparse.ArgumentParser()
for prm in paramNames:
    parser.add_argument('--'+prm)

args = parser.parse_args()

paramsDict = {prm: getattr(args,prm) for prm in paramNames}

agent = Proxy(computerName[args.rigName] + ':5000')
agent.start_script(script=runTaskPath, params=paramsDict)

stop = raw_input("type 'stop' and enter to terminate script: ")
if stop == 'stop':
    agent.stop_script()
