# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:55:21 2026

@author: samg
"""

import copy
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn.metrics
import sklearn.cluster
from DynamicRoutingAnalysisUtils import getPerformanceStats,getIsStandardRegimen,getFirstExperimentSession,getSessionsToPass,getSessionData,pca,cluster,fitCurve,calcWeibullDistrib


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

isStandardRegimen = getIsStandardRegimen(summaryDf)

hitThresh = 100
dprimeThresh = 1.5
nInitialTrainingSessions = 4

deltaLickProbLabels = ('5 rewarded targets',
                       '5 non-rewarded targets',
                       '1 rewarded target',
                       '1 non-rewarded target',
                       '5 rewards',
                       '5 catch trials')
deltaLickProb = {lbl: {targ: np.nan for targ in ('rewTarg','nonRewTarg')} for lbl in deltaLickProbLabels}


def getBlockTrials(obj,block,epoch):
    blockTrials = (obj.trialBlock==block) & ~obj.autoRewardScheduled
    n = blockTrials.sum()
    half = int(n/2)
    startTrial = half if epoch=='last half' else 0
    endTrial = half if epoch=='first half' else n
    return np.where(blockTrials)[0][startTrial:endTrial]


def detrend(r,order=2):
    x = np.arange(r.size)
    return r - np.polyval(np.polyfit(x,r,order),x)


def getCorrelation(r1,r2,rs1,rs2,corrSize=200,detrendOrder=None):
    if detrendOrder is not None:
        r1 = detrend(r1,detrendOrder)
        r2 = detrend(r2,detrendOrder)
        rs1 = rs1.copy()
        rs2 = rs2.copy()
        for z in range(rs1.shape[1]):
            rs1[:,z] = detrend(rs1[:,z],detrendOrder)
            rs2[:,z] = detrend(rs2[:,z],detrendOrder)
    c = np.correlate(r1,r2,'full') / (np.linalg.norm(r1) * np.linalg.norm(r2))   
    cs = np.mean([np.correlate(rs1[:,z],rs2[:,z],'full') / (np.linalg.norm(rs1[:,z]) * np.linalg.norm(rs2[:,z])) for z in range(rs1.shape[1])],axis=0)
    n = c.size // 2 + 1
    corrRaw = np.full(corrSize,np.nan)
    corrRaw[:n] = c[-n:]
    corr = np.full(corrSize,np.nan)
    corr[:n] = (c-cs)[-n:] 
    return corr,corrRaw




baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Sam\behav_spreadsheet_copies\has_licks"

for fileName in ('DynamicRoutingTrainingNSB.xlsx',):
    failures = []
    
    excelPath = os.path.join(baseDir,fileName)
    sheets = pd.read_excel(excelPath,sheet_name=None)
    writer =  pd.ExcelWriter(excelPath,mode='a',engine='openpyxl',if_sheet_exists='replace',datetime_format='%Y%m%d_%H%M%S')
    allMiceDf = sheets['all mice']
    mouseIds = allMiceDf['mouse id']
    for mi,mouseId in enumerate(mouseIds):
        print(str(mi+1)+' of '+str(len(mouseIds)))
        if str(mouseId) in sheets:
            df = sheets[str(mouseId)]
            hasLicks = []
            for startTime in df['start time']:
                try:
                    obj = getSessionData(mouseId,startTime,lightLoad=True)
                    hasLicks.append(1 if len(obj.lickFrames)>0 else 0)
                except:
                    failures.append((mouseId,startTime))
                    hasLicks.append(1)
            df.rename(columns={'ignore':'has licks'},inplace=True)
            df['has licks'] = hasLicks
            
            df.to_excel(writer,sheet_name=str(mouseId),index=False)
            sheet = writer.sheets[str(mouseId)]
            if 'NSB' in fileName:
                for col in ('ABCDEFGHIJK'):
                    if col in ('I','J','K','L'):
                        w = 10
                    elif col in ('B','C','H'):
                        w = 15
                    elif col=='D':
                        w = 40
                    else:
                        w = 30
                    sheet.column_dimensions[col].width = w
            else:
                for col in ('ABCDEFGHIJK'):
                    if col in ('H','I','J','K','L'):
                        w = 10
                    elif col in ('B','G'):
                        w = 15
                    elif col=='C':
                        w = 40
                    else:
                        w = 30
                    sheet.column_dimensions[col].width = w
    
    allMiceDf.to_excel(writer,sheet_name='all mice',index=False)
    sheet = writer.sheets['all mice']
    if 'NSB' in fileName:
        for col in ('ABCDEFGHIJKLMNOPQR'):
            if col == 'G':
                w = 20
            elif col == 'R':
                w = 30
            else:
                w = 12
            sheet.column_dimensions[col].width = w
    else:
        for col in ('ABCDEFGHIJKLMNOPQR'):
            if col == 'G':
                w = 20
            elif col == 'R':
                w = 30
            else:
                w = 12
            sheet.column_dimensions[col].width = w
    # writer.save()
    writer.close()




summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

isStandardRegimen = getIsStandardRegimen(summaryDf)

mice = {'stage 1 pass': np.array(summaryDf[isStandardRegimen & summaryDf['stage 1 pass']]['mouse id'])}
sessionsToPass = []
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.where(np.array(['stage 1' in task for task in df['task version']]) & np.array(df['has licks'].astype(bool)))[0]
        sessionsToPass.append(getSessionsToPass(mid,df,sessions,stage=1))
        
mice = {'stage 2 pass': np.array(summaryDf[isStandardRegimen & summaryDf['stage 2 pass']]['mouse id'])}
sessionsToPass = []
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.where(np.array(['stage 2' in task for task in df['task version']]) & np.array(df['has licks'].astype(bool)))[0]
        sessionsToPass.append(getSessionsToPass(mid,df,sessions,stage=2))

mice = {'stage 5 pass': np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass']]['mouse id'])}
sessionsToPass = []
for lbl in mice:
    for mid in mice[lbl]:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']]) & np.array(df['has licks'].astype(bool))
        firstExperimentSession = getFirstExperimentSession(df)
        if firstExperimentSession is not None:
            sessions[firstExperimentSession:] = False
        sessions = np.where(sessions)[0]
        sessionsToPass.append(getSessionsToPass(mid,df,sessions,stage=5))


























