# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:36:05 2021

@author: svc_ccg
"""

import argparse
from zro import Proxy

computer_name = {'NP3': 'w10DTSM118296',
                 'E1': 'wxvs-syslogic31',
                 'E2': 'wxvs-syslogic32',
                 'E3': 'wxvs-syslogic33',
                 'E4': 'wxvs-syslogic34',
                 'E5': 'wxvs-syslogic35',
                 'E6': 'wxvs-syslogic36'}

runTaskPath = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\DynamicRoutingTask\runTask.py'

#runTaskPath = r'C:\Users\svc_neuropix\Desktop\runTask.py'

parser = argparse.ArgumentParser()
parser.add_argument('--rigName', type=str,
                    help='name of rig')
parser.add_argument('--subjectName', type=str,
                    help='name of subject')
parser.add_argument('--taskScript', type=str,
                    help='path to task script')
parser.add_argument('--taskVersion', type=str,
                    help='version of task')

args = parser.parse_args()

params = {'rigName': args.rigName,
          'subjectName': args.subjectName,
          'taskScript': args.taskScript,
          'taskVersion': args.taskVersion}

camstim_agent = Proxy(computer_name[args.rigName] + ':5000')
camstim_agent.start_script(script=runTaskPath, params=params)

stop = raw_input("type 'stop' and enter to terminate script: ")
if stop == 'stop':
    camstim_agent.stop_script()