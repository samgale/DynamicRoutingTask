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

parser = argparse.ArgumentParser()
parser.add_argument('--rigName')
parser.add_argument('--mouseID',default=None)
parser.add_argument('--userName',default=None)
parser.add_argument('--lightOn',default=None)
parser.add_argument('--solenoidOpen',default=None)
args = parser.parse_args()

agent = Proxy(computerName[args.rigName] + ':5000')

if args.lightOn is not None:
    lightOn = args.lightOn == 'True'
    agent.light(lightOn)
elif args.solenoidOpen is not None:
    if args.solenoidOpen == 'True':
        agent.open_reward_line()
    else:
        agent.close_reward_line()
elif args.mouseID is not None and args.userName is not None:
    agent.start_session(args.mouseID, args.userName)
else:
    agent.stop_script()