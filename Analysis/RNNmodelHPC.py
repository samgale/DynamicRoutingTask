# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import argparse
import copy
import os
import pathlib
import random
import numpy as np
import pandas as pd
import torch
from DynamicRoutingAnalysisUtils import getFirstExperimentSession,getSessionsToPass,getSessionData



baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting')
        

if __name__ == "__main__":
    print(torch.cuda.is_available())
    
