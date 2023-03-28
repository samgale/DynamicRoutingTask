# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:36:05 2021

@author: svc_ccg
"""

import argparse
from zro import Proxy

computerName = {'NP2': 'w10DT713937',
				'NP3': 'w10DTSM118296'}

runTaskPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\runTask.py"

paramNames = ('rigName','subjectName','taskScript','taskVersion','galvoX','galvoY','optoAmp','optoDur')

parser = argparse.ArgumentParser()
for prm in paramNames:
    parser.add_argument('--'+prm)

args = parser.parse_args()

paramsDict = {prm: getattr(args,prm) for prm in paramNames}

agent = Proxy(computerName[args.rigName] + ':5000')
agent.start_script(script=runTaskPath, params=paramsDict)
