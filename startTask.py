# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:36:05 2021

@author: svc_ccg
"""

import argparse
from zro import Proxy

computerName = {'NP3': 'w10DTSM118296',
                'B1': 'wxvs-syslogic7',
                'B2': 'wxvs-syslogic8',
                'B3': 'wxvs-syslogic9',
                'B4': 'wxvs-syslogic10',
                'B5': 'wxvs-syslogic11',
                'B6': 'wxvs-syslogic12'}

runTaskPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\runTask.py"

paramNames = ('rigName','subjectName','taskScript','taskVersion')

parser = argparse.ArgumentParser()
for prm in paramNames:
    parser.add_argument('--'+prm)

args = parser.parse_args()

paramsDict = {prm: getattr(args,prm) for prm in paramNames}

agent = Proxy(computerName[args.rigName] + ':5000')
agent.start_script(script=runTaskPath, params=paramsDict)
