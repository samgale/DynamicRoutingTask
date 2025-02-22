# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:37:15 2024

@author: svc_ccg
"""

import glob
import os
import numpy as np
import pandas as pd
from DynamicRoutingAnalysisUtils import getSessionsToPass


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"


summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

miceToIgnore = summaryDf['wheel fixed'] | summaryDf['cannula']

hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])

ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['stage 5 repeats'] & ~miceToIgnore
miceToUse = tuple(summaryDf[ind]['mouse id'])

nonStandardTrainingMice = (644864,644866,644867,681532,686176)
miceToUse += nonStandardTrainingMice


miceWithAllVideos = []
miceWithSomeVideos = []
for m in miceToUse:
    videoPath = os.path.join(baseDir,'behaviorvideos',str(m))
    if len(glob.glob(videoPath)) > 0:
        df = drSheets[str(m)] if str(m) in drSheets else nsbSheets[str(m)]
        sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
        sessions = np.where(sessions)[0]
        sessionsToPass = getSessionsToPass(m,df,sessions,stage=5)
        for st in df.loc[sessions[:sessionsToPass],'start time']:
            startDate = st.strftime('%Y%m%d')
            v = glob.glob(os.path.join(videoPath,str(m)+'*'+startDate+'*.mp4'))
            if len(v) == 0:
                break
        else:
            miceWithAllVideos.append((str(m),sessionsToPass,startDate))
        if (str(m),sessionsToPass) not in miceWithAllVideos:
            miceWithSomeVideos.append((str(m),sessionsToPass))
            
print(miceWithAllVideos)
        






