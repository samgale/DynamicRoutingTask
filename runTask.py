# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:48:22 2021

@author: svc_ccg
"""

import sys
import subprocess

env = 'DynamicRoutingTaskDev'

taskScript, paramsPath = sys.argv

toRun = """
call activate env
python {taskScript} {paramsPath}
"""

p = subprocess.Popen(toRun)