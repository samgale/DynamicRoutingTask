# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:38:41 2025

@author: svc_ccg
"""

import glob
import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getSessionsToPass,getSessionData,getPerformanceStats
from RNNmodelHPC import getRNNSessions


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"


def boxcar(data,smoothSamples):
    smoothFilter = np.ones(smoothSamples) / smoothSamples
    smoothedData = np.convolve(data,smoothFilter,mode='same')
    smoothedData[:smoothSamples] = smoothedData[smoothSamples]
    smoothedData[-smoothSamples:] = smoothedData[-smoothSamples]
    return smoothedData


#
summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
isStandardRegimen = getIsStandardRegimen(summaryDf)
mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])
sessionStartTimes = []

for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = getRNNSessions(df)
    sessionStartTimes.append([st.strftime('%Y%m%d_%H%M%S') for st in df.loc[sessions,'start time']])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
dsort = np.sort([len(s) for s in sessionStartTimes])
cumProb = np.array([np.sum(dsort>=i)/dsort.size for i in dsort])
ax.plot(dsort,cumProb,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,dsort[-1]+1])
ax.set_ylim([0,1.01])
ax.set_xlabel('# sessions')
ax.set_ylabel('Cumalative fraction of mice',fontsize=16)
plt.tight_layout()

maxTrainSessions = 20
mouseIds = []
for mouseId in mice:
    df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
    sessions = getRNNSessions(mouseId,df)
    if len(sessions) > maxTrainSessions:
        mouseIds.append(mouseId)


#
modelData = {}
filePaths = glob.glob(os.path.join(baseDir,'Sam','RNNmodel','*.npz'))
for f in filePaths:
    fileParts = os.path.splitext(os.path.basename(f))[0].split('_')
    mouseId,sessionDate,sessionTime,hiddenType,nTrainSessions,nHiddenUnits = fileParts
    session = sessionDate+'_'+sessionTime
    nTrainSessions = int(nTrainSessions[:nTrainSessions.find('train')])
    nHiddenUnits = int(nHiddenUnits[:nHiddenUnits.find('hidden')])
    if mouseId not in modelData:
        modelData[mouseId] = {}
    if session not in modelData[mouseId]:
        modelData[mouseId][session] = {}
    if hiddenType not in modelData[mouseId][session]:
        modelData[mouseId][session][hiddenType] = {}
    if nTrainSessions not in modelData[mouseId][session][hiddenType]:
        modelData[mouseId][session][hiddenType][nTrainSessions] = {}
    d = modelData[mouseId][session][hiddenType][nTrainSessions][nHiddenUnits] = {}
    with np.load(f,allow_pickle=True) as data:
        for key in data.keys():
            d[key] = data[key]
    
      
hiddenTypes = ('rnn','gru','lstm')
nTrainSessions = np.array([4,8,12,16,20])
nHiddenUnits = np.array([2,4,8,16,32])         
 
for mouseId in modelData:
    for session in modelData[mouseId]:
        for hiddenType in hiddenTypes:
            fig = plt.figure(figsize=(12,10))
            gs = matplotlib.gridspec.GridSpec(nHiddenUnits.size,nTrainSessions.size)
            for i,nh in enumerate(nHiddenUnits[::-1]):
                for j,nt in enumerate(nTrainSessions):
                    ax = fig.add_subplot(gs[i,j])
                    d = modelData[mouseId][session][hiddenType][nt][nh]
                    ax.plot(d['logLossTrain'],'r')
                    ax.plot(d['logLossTest'],'b')
                    for side in ('right','top'):
                        ax.spines[side].set_visible(False)
                    ax.tick_params(direction='out')
                    ax.set_xlim([0,30000])
                    ax.set_ylim([0,1])
            plt.tight_layout()


for mouseId in modelData:
    for session in modelData[mouseId]:
        for hiddenType in hiddenTypes:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            logLoss = np.zeros((nHiddenUnits.size,nTrainSessions.size))
            for i,nh in enumerate(nHiddenUnits[::-1]):
                for j,nt in enumerate(nTrainSessions):
                    testLoss = modelData[mouseId][session][hiddenType][nt][nh]['logLossTest']
                    bestIter = np.nanargmin(testLoss)
                    logLoss[i,j] = np.mean(testLoss[bestIter-10:bestIter+10])
            im = ax.imshow(logLoss,cmap='magma',clim=(0.2,0.5))
            cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
            # cb.set_ticks((0,0.2,0.4,0.6))
            # cb.set_ticklabels((0,0.2,0.4,0.6),fontsize=12)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out')
            ax.set_xticks(np.arange(nTrainSessions.size))
            ax.set_yticks(np.arange(nHiddenUnits.size))
            ax.set_xticklabels(nTrainSessions)
            ax.set_yticklabels(nHiddenUnits[::-1])
            ax.set_xlabel('# Training Sessions')
            ax.set_ylabel('# Hidden Units')
            ax.set_title('-log(likelihood)')
            plt.tight_layout()


#
nSessions = sum([len(modelData[mouseId]) for mouseId in modelData])
for hiddenType in hiddenTypes:
    logLoss = np.zeros((nSessions,nHiddenUnits.size,nTrainSessions.size))
    k = 0
    for mouseId in modelData:
        for session in modelData[mouseId]:
            for i,nh in enumerate(nHiddenUnits[::-1]):
                for j,nt in enumerate(nTrainSessions):
                    testLoss = modelData[mouseId][session][hiddenType][nt][nh]['logLossTest']
                    bestIter = np.nanargmin(testLoss)
                    logLoss[k,i,j] = np.mean(testLoss[bestIter-10:bestIter+10])
            k += 1
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(np.mean(logLoss,axis=0),cmap='magma',clim=(0.2,0.5))
    cb = plt.colorbar(im,ax=ax,fraction=0.026,pad=0.04)
    # cb.set_ticks((0,0.2,0.4,0.6))
    # cb.set_ticklabels((0,0.2,0.4,0.6),fontsize=12)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out')
    ax.set_xticks(np.arange(nTrainSessions.size))
    ax.set_yticks(np.arange(nHiddenUnits.size))
    ax.set_xticklabels(nTrainSessions)
    ax.set_yticklabels(nHiddenUnits[::-1])
    ax.set_xlabel('# Training Sessions')
    ax.set_ylabel('# Hidden Units')
    ax.set_title('-log(likelihood)')
    plt.tight_layout()






