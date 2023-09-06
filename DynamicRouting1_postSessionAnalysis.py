# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:19:05 2023

@author: svc_ccg
"""

import sys
import h5py
from DynamicRoutingAnalysisUtils import DynRoutData


def postSessionAnalysis(filePath):
    obj = DynRoutData()
    obj.loadBehavData(filePath)
    
    with h5py.File(filePath,'r+') as f:
        analysis = f.create_group('analysis')
        for key in ('blockStimRewarded','hitCount','dprimeSameModal','dprimeOtherModalGo'):
            analysis.create_dataset(key,data=getattr(obj,key))

            
if __name__ == "__main__":
    filePath = sys.argv[1]
    postSessionAnalysis(filePath)