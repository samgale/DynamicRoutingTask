# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:38:37 2024

@author: svc_ccg
"""

import copy
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import DynRoutData


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"

decodeDataPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\CO decoding results\2024-10-11\decoder_confidence_versus_trials_since_rewarded_target_all_units.pkl"

df = pd.read_pickle(decodeDataPath)

areaNames = np.unique(df['area'])

areas = ('ORBl','ORBm','ORBvl') + ('ACAd','ACAv') + ('PL',) + ('MOs',) + ('SCig','SCiw','SCdg','MRN')
sessions = np.in1d(df['area'],areas)
nSessions = sessions.sum()


# intra-block resp rate correlations
stimNames = ('vis1','sound1','vis2','sound2','decoder')
autoCorr = [[] for _ in range(5)]
corrWithin = [[[] for _ in range(5)] for _ in range(5)]
corrAcross = copy.deepcopy(corrWithin)
autoCorrMat = np.zeros((5,100))
corrWithinMat = np.zeros((5,5,200))
corrAcrossMat = copy.deepcopy(corrWithinMat)
nShuffles = 10
startTrial = 10

for sessionInd in np.where(sessions)[0]:
    print(sessionInd)
    sessionName = df['session'][sessionInd]
    fileName = 'DynamicRouting1_' + sessionName.replace('-','') + '*.hdf5'
    filePath = glob.glob(os.path.join(baseDir,'Data',sessionName[:6],fileName))
    obj = DynRoutData()
    obj.loadBehavData(filePath[0])
    
    decoderConf = np.full(obj.nTrials,np.nan)
    i = int(np.round(np.sum(obj.trialBlock==1)/2))
    c = df['confidence'][sessionInd]
    decoderConf[i:i+c.size] = c
        
    resp = np.zeros((5,obj.nTrials))
    respShuffled = np.zeros((5,obj.nTrials,nShuffles))
    for blockInd in range(6):
        blockTrials = np.where(obj.trialBlock==blockInd+1)[0][startTrial:]
        for i,s in enumerate(stimNames):
            if i < 4:
                stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0])
                r = obj.trialResponse[stimTrials].astype(float)
                r[r<1] = -1
            else:
                stimTrials = blockTrials
                r = decoderConf[stimTrials]
            resp[i,stimTrials] = r
            for z in range(nShuffles):
                respShuffled[i,stimTrials,z] = np.random.permutation(r)
    
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
        if blockInd in (0,5):
            continue
        blockTrials = np.where(obj.trialBlock==blockInd+1)[0][startTrial:]
        for i,s in enumerate(stimNames if rewStim=='vis1' else ('sound1','vis1','sound2','vis2','decoder')):
            stimTrials = np.intersect1d(blockTrials,np.where(obj.trialStim==s)[0]) if i < 4 else blockTrials
            if len(stimTrials) < 1:
                continue
            r = obj.trialResponse[stimTrials].astype(float) if i < 4 else decoderConf[stimTrials]
            c = np.correlate(r,r,'full')
            norm = np.linalg.norm(r)**2
            cc = []
            for _ in range(nShuffles):
                rs = np.random.permutation(r)
                cs = np.correlate(rs,rs,'full')
                cc.append(c - cs)
                cc[-1] /= norm
            n = c.size // 2
            a = np.full(100,np.nan)
            a[:n] = np.mean(cc,axis=0)[-n:]
            autoCorr[i].append(a)
        
        r = resp[:,blockTrials]
        rs = respShuffled[:,blockTrials]
        if rewStim == 'sound1':
            r = r[[1,0,3,2,4]]
            rs = rs[[1,0,3,2,4]]
        for i,(r1,rs1) in enumerate(zip(r,rs)):
            for j,(r2,rs2) in enumerate(zip(r,rs)):
                if len(r1) < 1 or len(r2) < 1:
                    continue
                c = np.correlate(r1,r2,'full')
                norm = np.linalg.norm(r1) * np.linalg.norm(r2)
                cc = []
                for z in range(nShuffles):
                    cs = np.correlate(rs1[:,z],rs2[:,z],'full')
                    cc.append(c - cs)
                    cc[-1] /= norm
                n = c.size // 2
                a = np.full(200,np.nan)
                a[:n] = np.mean(cc,axis=0)[-n:]
                corrWithin[i][j].append(a)
        
        otherBlocks = [0,2,4] if blockInd in [0,2,4] else [1,3,5]
        otherBlocks.remove(blockInd)
        a = np.full((2,200),np.nan)
        for k,b in enumerate(otherBlocks):
            bTrials = np.where(obj.trialBlock==b+1)[0][startTrial:]
            rOther = resp[:,bTrials]
            rsOther = respShuffled[:,bTrials]
            if rewStim == 'sound1':
                rOther = rOther[[1,0,3,2]]
                rsOther = rsOther[[1,0,3,2]]
            for i,(r1,rs1) in enumerate(zip(rOther,rsOther)):
                for j,(r2,rs2) in enumerate(zip(r,rs)):
                    c = np.correlate(r1,r2,'full')
                    norm = np.linalg.norm(r1) * np.linalg.norm(r2)
                    cc = []
                    for z in range(nShuffles):
                        cs = np.correlate(rs1[:,z],rs2[:,z],'full')
                        cc.append(c - cs)
                        cc[-1] /= norm
                    n = c.size // 2
                    a = np.full(200,np.nan)
                    a[:n] = np.mean(cc,axis=0)[-n:]
                    corrAcross[i][j].append(a)
                


stimLabels = ('rewarded target','unrewarded target','non-target\n(rewarded modality)','non-target\n(unrewarded modality)','decoder')


fig = plt.figure(figsize=(4,6))           
gs = matplotlib.gridspec.GridSpec(5,1)
x = np.arange(100) + 1
for i,lbl in enumerate(stimLabels):
    ax = fig.add_subplot(gs[i])
    m = np.nanmean(autoCorr[i],axis=0)
    s = np.nanstd(autoCorr[i],axis=0) / (len(autoCorr[i]) ** 0.5)
    ax.plot(x,m,'k')
    ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,20,5))
    ax.set_xlim([0,15])
    ax.set_ylim([-0.06,0.12])
    if i==4:
        ax.set_xlabel('Lag (trials)')
    if i==0:
        ax.set_ylabel('Auto-correlation')
    ax.set_title(lbl)
plt.tight_layout()

for mat in (corrWithin,):
    fig = plt.figure(figsize=(10,8))          
    gs = matplotlib.gridspec.GridSpec(5,5)
    x = np.arange(200) + 1
    for i,ylbl in enumerate(stimLabels):
        for j,xlbl in enumerate(stimLabels):
            ax = fig.add_subplot(gs[i,j])
            # for y in mat[i,j]:
            #     ax.plot(x,y,'k',alpha=0.2)
            m = np.nanmean(mat[i][j],axis=0)
            s = np.nanstd(mat[i][j],axis=0) / (len(mat[i][j]) ** 0.5)
            ax.plot(x,m,'k')
            ax.fill_between(x,m-s,m+s,color='k',alpha=0.25)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=9)
            ax.set_xlim([0,30])
            ax.set_ylim([-0.025,0.075])
            if i==4:
                ax.set_xlabel('Lag (trials)',fontsize=11)
            if j==0:
                ax.set_ylabel(ylbl,fontsize=11)
            if i==0:
                ax.set_title(xlbl,fontsize=11)
    plt.tight_layout()





