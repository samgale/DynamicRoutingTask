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
parser.add_argument('--rigName',type=str)
parser.add_argument('--lightOn',type=bool)
args = parser.parse_args()

agent = Proxy(computerName[args.rigName] + ':5000')
agent.light(args.lightOn)