# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import argparse
import itertools
import os
import numpy as np
import DynamicRoutingAnalysisUtils


baseDir = '/allen/programs/braintv/workgroups/tiny-blue-dot/masking/Sam'

def findBestFit(jobInd,totalJobs):
    
    dataPath = os.path.join(baseDir,'Analysis')

    bestFitParams = None
    bestFitError = None
    for fitParams in itertools.islice(fitParamsIter,paramsStart,paramsStart+paramCombosPerJob):
        modelError = calcModelError(fitParams,*fixedParams)
        if bestFitError is None or modelError < bestFitError:
            bestFitParams = fitParams
            bestFitError = modelError
    np.savez(os.path.join(baseDir,'HPC','fit_'+str(jobInd)+'.npz'),params=bestFitParams,error=bestFitError)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobInd',type=int)
    parser.add_argument('--totalJobs',type=int)
    args = parser.parse_args()
    findBestFit(args.jobInd,args.totalJobs)
