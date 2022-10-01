# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:55:44 2021

@author: svc_ccg
"""

import copy
import glob
import os
import re
import h5py
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import fileIO
from DynamicRoutingAnalysisUtils import DynRoutData,sortExps,updateTrainingStage,makeSummaryPdf
from DynamicRoutingAnalysisUtils import fitCurve,calcLogisticDistrib,calcWeibullDistrib,inverseLogistic,inverseWeibull


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"


# update training spreadsheet
updateTrainingStage(replaceData=False)
    
   
# get data
behavFiles = []
while True:
    files = fileIO.getFiles('choose experiments',rootDir=os.path.join(baseDir,'Data'),fileType='*.hdf5')
    if len(files)>0:
        behavFiles.extend(files)
    else:
        break
    
if len(behavFiles)>0:
    exps = []
    for f in behavFiles:
        obj = DynRoutData()
        obj.loadBehavData(f)
        exps.append(obj)
        
# sort experiments by start time
exps = sortExps(exps)


# summary pdf
for obj in exps:
    makeSummaryPdf(obj)
    

# print summary
for obj in exps:
    print(obj.subjectName)
    for i,d in enumerate((obj.hitCount,obj.dprimeSameModal,obj.dprimeOtherModalGo)):
        if i>0:
            d = np.round(d,2)
        print(*d,sep=', ')
    print('\n')
    

# smoothed resp prob
fig = plt.figure(figsize=(12,10))
ylim = [-0.05,1.05]
smoothSigma = 5
for i,obj in enumerate(exps):
    ax = fig.add_subplot(len(exps),1,i+1)
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        blockTrials = obj.trialBlock==blockInd+1
        blockStart,blockEnd = np.where(blockTrials)[0][[0,-1]]
        if goStim=='vis1':
            lbl = 'vis rewarded' if blockInd==0 else None
            ax.add_patch(matplotlib.patches.Rectangle([blockStart+0.5,ylim[0]],width=blockEnd-blockStart+1,height=ylim[1]-ylim[0],facecolor='0.8',edgecolor=None,alpha=0.2,zorder=0,label=lbl))
        for stim,clr,ls in zip(('vis1','vis2','sound1','sound2'),'ggmm',('-','--','-','--')):
            trials = blockTrials & (obj.trialStim==stim) #& ~obj.autoRewarded
            smoothedRespProb = scipy.ndimage.gaussian_filter(obj.trialResponse[trials].astype(float),smoothSigma)
            lbl = stim if i==0 and blockInd==0 else None
            ax.plot(np.where(trials)[0]+1,smoothedRespProb,color=clr,ls=ls,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlim([0.5,blockEnd+1.5])
    ax.set_ylim(ylim)
    if i==len(exps)-1:
        ax.set_xlabel('trial',fontsize=12)
    if i==0:
        ax.set_ylabel('resp prob',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1.5),fontsize=8)
    ax.set_title(obj.subjectName+'_'+obj.startTime,fontsize=10)
plt.tight_layout()


# training summary
excelPath = os.path.join(baseDir,'DynamicRoutingTraining.xlsx')
sheets = pd.read_excel(excelPath,sheet_name=None)
allMiceDf = sheets['all mice']
mouseIds = allMiceDf['mouse id']
regimen = allMiceDf['regimen']
craniotomy = allMiceDf['craniotomy']

for stage in ('stage 1','stage 2'):
    running = []
    timeouts = []
    long = []
    moving = []
    passInd = []
    passHits = []
    passDprime = []
    reg1PassInd = []
    fig,axs = plt.subplots(2)
    fig.set_size_inches(8,6)
    xmax = 0
    for mid in mouseIds:
        if str(mid) in sheets:
            mouseInd = np.where(allMiceDf['mouse id']==mid)[0][0]
            if craniotomy[mouseInd]:
                pass#continue
            df = sheets[str(mid)]
            
            # sessions = np.array([(stage in task and not 'templeton' in task) for task in df['task version']])
            sessions = np.array([stage in task for task in df['task version']])
            nSessions = np.sum(sessions)
            if nSessions==0:
                continue
            running.append(not allMiceDf.loc[mouseInd,'wheel fixed'])
            timeouts.append(allMiceDf.loc[mouseInd,'timeouts'])
            long.append(np.any(['long' in task for task in df['task version']]))
            moving.append(np.any(['moving' in task for task in df['task version']]))
            passInd.append(np.nan)
            passHits.append(np.nan)
            passDprime.append(np.nan)
            reg1PassInd.append(np.nan)
            hits = np.array([int(re.findall('[0-9]+',s)[0]) for s in df[sessions]['hits']])
            dprime = np.array([float(re.findall('-*[0-9].[0-9]*',s)[0]) for s in df[sessions]['d\' same modality']])
            for i in range(nSessions):
                if i > 0:
                    if all(hits[i-1:i+1] > 100) and all(dprime[i-1:i+1] > 1.5):
                        passInd[-1] = i
                        passHits[-1] = hits[i]
                        passDprime[-1] = dprime[i]
                        break
            else:
                if any(str(int(stage[-1])+1) in task for task in df['task version']):
                    passInd[-1] = nSessions-1
                    passHits[-1] = hits[passInd[-1]]
                    passDprime[-1] = dprime[passInd[-1]]
            if regimen[mouseInd]==1:
                for i in range(nSessions):
                    if i > 0:
                        if all(hits[i-1:i+1] > 150) and all(dprime[i-1:i+1] > 1.5):
                            reg1PassInd[-1] = i
                            break
                    else:
                        if any(str(int(stage[-1])+1) in task for task in df['task version']):
                            reg1PassInd[-1] = nSessions-1
            x = np.arange(nSessions)+1
            xmax = max(xmax,nSessions+1.5)
            ls = '-' if running[-1] else '--'
            if moving[-1]:
                clr = 'k'
            elif timeouts[-1]:
                clr = 'm'
            else:
                clr = 'g'
            lw = 2 if long[-1] else 1
            lbl = 'run' if running[-1] else 'no run'
            lbl += ', timeouts' if timeouts[-1] else ', no timeouts'
            lbl += ', long' if long[-1] else ''
            lbl += ', moving' if moving[-1] else ''
            for ax,val in zip(axs,(hits,dprime)):
                if np.isnan(passInd[-1]):
                    ax.plot(x,val,color=clr,ls=ls,lw=lw,label=lbl)
                else:
                    ax.plot(x[:passInd[-1]+1],val[:passInd[-1]+1],color=clr,ls=ls,lw=lw,label=lbl)
                    ax.plot(passInd[-1]+1,val[passInd[-1]],'o',mec=clr,mfc='none')
    for i,(ax,ylbl,thresh) in enumerate(zip(axs,('hits','d prime'),(100,1.5))):
        ax.plot([0,xmax],[thresh]*2,'k:',zorder=0)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        xticks = np.arange(0,xmax+1,5) if xmax>11 else np.arange(xmax)
        ax.set_xticks(xticks)
        ax.set_xlim([0.5,xmax])
        if i==1:
            ax.set_xlabel('session')
        ax.set_ylabel(ylbl)
        if i==0:
            handles,labels = ax.get_legend_handles_labels()
            lblDict = dict(zip(labels,handles))
            ax.legend(lblDict.values(),lblDict.keys(),loc='lower right',fontsize=8)
            ax.set_title(stage)
    plt.tight_layout()
    
    passInd,passHits,passDprime,reg1PassInd,running,timeouts,long,moving = [np.array(d) for d in (passInd,passHits,passDprime,reg1PassInd,running,timeouts,long,moving)]
    passSession = passInd+1
    reg1PassSession = reg1PassInd+1
    
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # for r1,r2,run,to in zip(reg1PassSession,passSession,running,timeouts):
    #     if not np.isnan(r1):
    #         ls = '-' if run else '--'
    #         clr = 'm' if to else 'g'
    #         ax.plot([0,1],[r1,r2],'o-',color=clr,mfc='none',ls=ls)
    # ax.plot([0,1],[np.nanmedian(reg1PassSession),np.nanmedian(passSession)],'ko-',ms=10)
    # for side in ('right','top'):
    #     ax.spines[side].set_visible(False)
    # ax.tick_params(direction='out',top=False,right=False)
    # ax.set_xticks([0,1])
    # ax.set_xticklabels(['>150 hits\n(regimen 1)','>100 hits\n(regimen 2)'])
    # ax.set_xlim([-0.25,1.25])
    # ax.set_ylim([1,max(np.nanmax(reg1PassSession),np.nanmax(passSession))+1])
    # ax.set_ylabel('sessions to pass')
    # ax.set_title(stage+' (regimen 1 mice)')
    # plt.tight_layout()
    
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(4,1,1)
    notCompletedXtick = np.ceil(1.25*np.nanmax(passSession))
    for d,ls,mrk,lbl in zip((passSession[running & ~moving],passSession[~running & ~moving]),('-','--'),'os',('run, static','no run, static')):
        d[np.isnan(d)] = notCompletedXtick
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        lbl += ' (n='+str(d.size)+')'
        if len(cumProb)<2 or np.all(cumProb==1):
            ax.plot(dsort,cumProb,mrk,mec='k',mfc='none',mew=1,label=lbl)
        else:
            ax.plot(dsort,cumProb,color='k',ls=ls,lw=1,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    xticks = list(plt.get(ax,'xticks'))+[notCompletedXtick]
    xticklabels = xticks[:]
    xticklabels[-1] = 'not\ncompleted'
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    xlim = [np.nanmin(passSession)-0.5,notCompletedXtick+0.5]
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.02])
    ax.set_ylabel('cum. prob.')
    ax.legend(loc='lower right',fontsize=8)
    ax.set_title(stage)
    
    ax = fig.add_subplot(4,1,2)
    for d,clr,lbl in zip((passSession[timeouts & ~moving],passSession[~timeouts & ~moving]),'mg',('timeouts, static','no timeouts, static')):
        d[np.isnan(d)] = notCompletedXtick
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        lbl += ' (n='+str(d.size)+')'
        if len(cumProb)<2 or np.all(cumProb==1):
            ax.plot(dsort,cumProb,'o',mec=clr,mfc='none',mew=1,label=lbl)
        else:
            ax.plot(dsort,cumProb,color=clr,lw=1,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.02])
    ax.set_ylabel('cum. prob.')
    ax.legend(loc='lower right',fontsize=8)
    
    ax = fig.add_subplot(4,1,3)
    for d,lw,lbl in zip((passSession[long & ~moving],passSession[~long & ~moving]),(2,1),('long, static','not long, static')):
        d[np.isnan(d)] = notCompletedXtick
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        lbl += ' (n='+str(d.size)+')'
        if len(cumProb)<2 or np.all(cumProb==1):
            ax.plot(dsort,cumProb,'o',mec='k',mfc='none',mew=1,label=lbl)
        else:
            ax.plot(dsort,cumProb,'k',lw=lw,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.02])
    ax.set_ylabel('cum. prob.')
    ax.legend(loc='lower right',fontsize=8)
    
    ax = fig.add_subplot(4,1,4)
    for ind,clr,ls,lw,mrk,lbl in zip((running & timeouts & ~long & ~ moving,running & ~timeouts & ~long & ~ moving,~running & timeouts & ~long & ~moving,~running & ~timeouts & ~long & ~moving,long & timeouts & ~moving,long & ~timeouts & ~moving,moving),
                          'mgmgmgk',('-','-','--','--','-','-','-'),(1,1,1,1,2,2,1),'oossooo',
                          ('run, timeouts','run, no timeouts','no run, timeouts','no run, no timeouts','long, timeouts','long, no timeouts','moving')):
        d = passSession[ind]
        d[np.isnan(d)] = notCompletedXtick
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        lbl += ' (n='+str(d.size)+')'
        if len(cumProb)<2 or np.all(cumProb==1):
            ax.plot(dsort,cumProb,mrk,mec=clr,mfc='none',mew=lw,label=lbl)
        else:
            ax.plot(dsort,cumProb,color=clr,ls=ls,lw=lw,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('sessions to pass')
    ax.set_ylabel('cum. prob.')
    ax.legend(loc='lower right',fontsize=8)
    plt.tight_layout()
    
    fig = plt.figure(figsize=(6,8))
    for i,(param,xlbl) in enumerate(zip((passSession,passHits,passDprime),('sessions to pass','pass hits','pass dprime'))):
        ax = fig.add_subplot(3,1,i+1)
        for ind,clr,ls,lw,mrk,lbl in zip((running & timeouts & ~long & ~ moving,running & ~timeouts & ~long & ~ moving,~running & timeouts & ~long & ~moving,~running & ~timeouts & ~long & ~moving,long & timeouts & ~moving,long & ~timeouts & ~moving,moving),
                              'mgmgmgk',('-','-','--','--','-','-','-'),(1,1,1,1,2,2,1),'oossooo',
                              ('run, timeouts','run, no timeouts','no run, timeouts','no run, no timeouts','long, timeouts','long, no timeouts','moving')):
            d = param[ind]
            d = d[~np.isnan(d)]
            dsort = np.sort(d)
            cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
            lbl += ' (n='+str(d.size)+')'
            if len(cumProb)<2 or np.all(cumProb==1):
                ax.plot(dsort,cumProb,mrk,mec=clr,mfc='none',mew=lw,label=lbl)
            else:
                ax.plot(dsort,cumProb,color=clr,ls=ls,lw=lw,label=lbl)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticklabels)
        # ax.set_xlim(xlim)
        ax.set_ylim([0,1.02])
        ax.set_xlabel(xlbl)
        ax.set_ylabel('cum. prob.')
        if i==0:
            ax.legend(loc='lower right',fontsize=8)
        plt.tight_layout()
    
    
stage = 'stage 3'
for reg,hitThresh,substage in zip(((1,),(2,3,4),(2,)),(150,50,50),(1,1,2)):
    running = []
    timeouts = []
    long = []
    moving = []
    passInd = []
    fig,axs = plt.subplots(3)
    fig.set_size_inches(8,6)
    xmax = 0
    for mid in mouseIds:
        if str(mid) in sheets:
            mouseInd = np.where(allMiceDf['mouse id']==mid)[0][0]
            if regimen[mouseInd] not in reg:
                continue
            if craniotomy[mouseInd]:
                pass#continue
            df = sheets[str(mid)]
            sessions = np.array([(stage in task and not 'templeton' in task) for task in df['task version']])
            if substage==2:
                ind3 = np.where(['stage 3 tone' in task for task in df['task version']])[0]
                if len(ind3)==0:
                    continue
                else:
                    sessions[:ind3[0]] = False
                    ind4 = np.where(['stage 4' in task for task in df['task version']])[0]
                    if len(ind4)>0:
                        sessions[ind4[0]:] = False
            nextStage = 'stage 4' if 1 in reg or substage==2 else 'stage 3 tone'
            nextStageInd = np.where([nextStage in task for task in df['task version']])[0]
            if len(nextStageInd)>0:
                sessions[nextStageInd[0]:] = False
            nSessions = np.sum(sessions)
            if nSessions==0:
                continue
            running.append(not allMiceDf.loc[mouseInd,'wheel fixed'])
            timeouts.append(allMiceDf.loc[mouseInd,'timeouts'])
            long.append(np.any(['long' in task for task in df['task version']]))
            moving.append(np.any(['moving' in task for task in df['task version']]))
            passInd.append(np.nan)
            hits = np.array([int(re.findall('[0-9]+',s)[0]) for s in df[sessions]['hits']])
            dprimeSame = np.array([float(re.findall('-*[0-9].[0-9]*',s)[0]) for s in df[sessions]['d\' same modality']])
            if 1 in reg:
                dprimeOther = None
            else:
                dprimeOther = np.array([float(re.findall('-*[0-9].[0-9]*',s)[0]) for s in df[sessions]['d\' other modality go stim']]) 
            for i in range(nSessions):
                if i > 0 and all(hits[i-1:i+1] > hitThresh) and all(dprimeSame[i-1:i+1] > 1.5) and (1 in reg or all(dprimeOther[i-1:i+1] > 1.5)):
                    passInd[-1] = i
                    break
            x = np.arange(nSessions)+1
            xmax = max(xmax,nSessions+0.5)
            ls = '-' if running[-1] else '--'
            if moving[-1]:
                clr = 'k'
            elif timeouts[-1]:
                clr = 'm'
            else:
                clr = 'g'
            lw = 2 if long[-1] else 1
            lbl = 'run' if running[-1] else 'no run'
            lbl += ', timeouts' if timeouts[-1] else ', no timeouts'
            lbl += ', long' if long[-1] else ''
            lbl += ', moving' if moving[-1] else ''
            for ax,val in zip(axs,(hits,dprimeSame,dprimeOther)):
                if val is not None:
                    ax.plot(x,val,color=clr,ls=ls,label=lbl)
                    if not np.isnan(passInd[-1]):
                        ax.plot(passInd[-1]+1,val[passInd[-1]],'o',mec=clr,mfc='none')
    for i,(ax,ylbl,thresh) in enumerate(zip(axs,('hits','intra-modality d\'','inter-modality d\''),(hitThresh,1.5,1.5))):
        ax.plot([0,xmax],[thresh]*2,'k:',zorder=0)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        xticks = np.arange(0,xmax+1,5) if xmax>11 else np.arange(xmax)
        ax.set_xticks(xticks)
        ax.set_xlim([0.5,xmax])
        ylim = ax.get_ylim()
        ax.set_ylim([min(0,ylim[0]),ylim[1]])
        if (1 in reg and i==1) or (2 in reg and i==2):
            ax.set_xlabel('session')
        if 1 in reg and 'd\'' in ylbl:
            ylbl = 'd\''
        ax.set_ylabel(ylbl)
        if i==0:
            handles,labels = ax.get_legend_handles_labels()
            lblDict = dict(zip(labels,handles))
            ax.legend(lblDict.values(),lblDict.keys(),loc='lower right',fontsize=8)
            title = stage+', regimen '+str(reg)
            if 2 in reg:
                title += ', part '+str(substage)
            ax.set_title(title)
    if 1 in reg:
        fig.delaxes(axs[2])
    plt.tight_layout()
    
    passInd,running,timeouts,long,moving = [np.array(d) for d in (passInd,running,timeouts,long,moving)]
    passSession = passInd+1
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    notCompletedXtick = np.ceil(1.25*np.nanmax(passSession))
    for ind,clr,ls,lw,mrk,lbl in zip((running & timeouts & ~long & ~moving,running & ~timeouts & ~long & ~moving,~running & timeouts & ~long & ~moving,~running & ~timeouts & ~long & ~moving,long & timeouts & ~moving,long & ~timeouts & ~moving,moving),
                                     'mgmgmgk',('-','-','--','--','-','-','-'),(1,1,1,1,2,2,1),'oossooo',
                                     ('run, timeouts','run, no timeouts','no run, timeouts','no run, no timeouts','long, timeouts','long, no timeouts','moving')):
        d = passSession[ind]
        d[np.isnan(d)] = notCompletedXtick
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        lbl += ' (n='+str(d.size)+')'
        if len(cumProb)<2 or np.all(cumProb==1):
            ax.plot(dsort,cumProb,mrk,mec=clr,mfc='none',mew=lw,label=lbl)
        else:
            ax.plot(dsort,cumProb,color=clr,ls=ls,lw=lw,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    xticks = list(plt.get(ax,'xticks'))+[notCompletedXtick]
    xticklabels = xticks[:]
    xticklabels[-1] = 'not\ncompleted'
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    xlim = [np.nanmin(passSession)-0.5,notCompletedXtick+0.5]
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('sessions to pass')
    ax.set_ylabel('cum. prob.')
    ax.legend(loc='lower right',fontsize=8)
    plt.tight_layout()


stage = 'stage 4'
allReg = []
runningAllReg = []
timeoutsAllReg = []
longAllReg = []
passIndAllReg =[]
handoffMice4 = []
handoffSessionStartTimes4 = []
for reg in (1,2,3):
    nblocks = 3 if reg>2 else 2
    for version in ('blocks','modality'):
        if version=='modality' and reg==3:
            continue
        running = []
        timeouts = []
        long = []
        passInd = []
        stage4Mice = []
        dprimeCrossModal = []
        firstBlockVis = []
        fig,axs = plt.subplots(3,nblocks)
        fig.set_size_inches(8,6)
        fig.suptitle(stage+', regimen '+str(reg))
        xmax = 0
        for mid in mouseIds:
            if str(mid) in sheets:
                mouseInd = np.where(allMiceDf['mouse id']==mid)[0][0]
                if regimen[mouseInd]!=reg:
                    continue
                if craniotomy[mouseInd]:
                    pass#continue
                df = sheets[str(mid)]
                sessions = np.array([(stage in task and not 'templeton' in task) for task in df['task version']])
                nSessions = np.sum(sessions)
                if nSessions==0:
                    continue
                if version=='blocks':
                    allReg.append(reg)
                running.append(not allMiceDf.loc[mouseInd,'wheel fixed'])
                timeouts.append(allMiceDf.loc[mouseInd,'timeouts'])
                long.append(np.any(['long' in task for task in df['task version']]))
                passInd.append(np.nan)
                oriFirst = np.array(['ori tone' in task for task in df[sessions]['task version']])
                hits = np.array([[float(s) for s in re.findall('[0-9]+',d)] for d in df[sessions]['hits']])
                dprimeSame = np.array([[float(s) for s in re.findall('-*[0-9nan].[0-9nan]*',d)] for d in df[sessions]['d\' same modality']])
                dprimeOther = np.array([[float(s) for s in re.findall('-*[0-9nan].[0-9nan]*',d)] for d in df[sessions]['d\' other modality go stim']])
                hits[np.isnan(dprimeOther)] = np.nan
                stage4Mice.append(mid)
                dprimeCrossModal.append(dprimeOther)
                firstBlockVis.append(oriFirst)
                for i in range(nSessions):
                    if i > 0 and np.all(dprimeSame[i-1:i+1] > 1.5) and np.all(dprimeOther[i-1:i+1] > 1.5):
                        passInd[-1] = i
                        if nblocks==2 and version=='blocks':
                            handoffMice4.append(str(mid))
                            handoffSessionStartTimes4.append(list(df['start time'][sessions][i-1:i+1]))
                        break
                x = np.arange(nSessions)+1
                xmax = max(xmax,nSessions+0.5)
                ls = '-' if running[-1] else '--'
                clr = 'm' if timeouts[-1] else 'g'
                lbl = 'run' if running[-1] else 'no run'
                lbl += ', timeouts' if timeouts[-1] else ', no timeouts'
                for i,val in enumerate((hits,dprimeSame,dprimeOther)):
                    for j in range(nblocks):
                        if j>1 and reg<3:
                            break
                        if version=='blocks' or reg==3:
                            v = val[:,j]
                        elif j==0:
                            v = val[np.stack((oriFirst,~oriFirst),axis=-1)]
                        else:
                            v = val[np.stack((~oriFirst,oriFirst),axis=-1)]
                        axs[i,j].plot(x,v,color=clr,ls=ls,label=lbl)
                        if not np.isnan(passInd[-1]):
                            axs[i,j].plot(passInd[-1]+1,v[passInd[-1]],'o',mec=clr,mfc='none')
        for i,ylbl in enumerate(('hits','intra-modality d\'','inter-modality d\'')):
            for j in range(nblocks):
                ax = axs[i,j]
                if i>0:
                    ax.plot([0,xmax],[1.5]*2,'k:',zorder=0)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xlim([0.5,xmax])
                ylim = ax.get_ylim()
                ax.set_ylim([min(0,ylim[0]),ylim[1]])
                if i==2:
                    ax.set_xlabel('session')
                if j==0:
                    ax.set_ylabel(ylbl)
                if i==0:
                    if j==0:
                        title = 'block 1' if version=='blocks' and reg<3 else 'vis block'
                        ax.set_title(title)
                        handles,labels = ax.get_legend_handles_labels()
                        lblDict = dict(zip(labels,handles))
                        ax.legend(lblDict.values(),lblDict.keys(),loc='lower right',fontsize=8)
                    elif j==1:
                        title = 'block 2' if version=='blocks' and reg<3 else 'sound block'
                        ax.set_title(title)
                    elif j==2 and reg==3:
                        ax.set_title('vis block')
        plt.tight_layout()
        
        if version=='blocks':
            runningAllReg.extend(running)
            timeoutsAllReg.extend(timeouts)
            longAllReg.extend(long)
            passIndAllReg.extend(passInd)

    fig = plt.figure(figsize=(16,8))
    fig.suptitle('Stage 4, regimen '+str(reg)+', '+'inter-modality d\'')
    nMice = len(dprimeCrossModal)
    for ind,(d,mid,vis,pi) in enumerate(zip(dprimeCrossModal,stage4Mice,firstBlockVis,passInd)):
        if not np.isnan(pi):
            d = d[:pi+1]
            vis = vis[:pi+1]
        nSessions,nBlocks = d.shape
        ax = fig.add_subplot(1,nMice,ind+1)
        cmax = np.nanmax(np.absolute(d))
        im = ax.imshow(d,cmap='bwr',clim=(-cmax,cmax))
        for i in range(nSessions):
            for j in range(nBlocks):
                ax.text(j,i,str(round(d[i,j],2)),ha='center',va='center',fontsize=6)
        ax.set_xticks(np.arange(nBlocks))
        ax.set_xticklabels(np.arange(nBlocks)+1)
        yticks = np.arange(nSessions) if nSessions<10 else np.concatenate(([0],np.arange(4,nSessions,5)))
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks+1)
        ax.set_ylim([nSessions-0.5,-0.5])
        ax.set_xlabel('block')
        if ind==0:
            ax.set_ylabel('session')
        for y,v in enumerate(vis):
            lbl = ''
            if v and reg<3:
                lbl += 'vis first'
            if y==pi:
                if v:
                    lbl += ', '
                lbl += '*pass*'
            ax.text(nBlocks-0.4,y,lbl,ha='left',va='center',fontsize=8)
        ax.set_title(mid)
    plt.tight_layout()

reg,passInd,running,timeouts,long = [np.array(d) for d in (allReg,passIndAllReg,runningAllReg,timeoutsAllReg,longAllReg)]
passSession = passInd+1

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
notCompletedXtick = np.ceil(1.25*np.nanmax(passSession))
for ind,clr,ls,lw,mrk,lbl in zip((running & timeouts,running & ~timeouts,~running & timeouts,~running & ~timeouts),
                      'mgmg',('-','-','--','--'),(1,1,1,1),'ooss',
                      ('run, timeouts','run, no timeouts','no run, timeouts','no run, no timeouts')):
    d = passSession[ind]
    d[np.isnan(d)] = notCompletedXtick
    dsort = np.sort(d)
    cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
    lbl += ' (n='+str(d.size)+')'
    if len(cumProb)<2 or np.all(cumProb==1):
        ax.plot(dsort,cumProb,mrk,mec=clr,mfc='none',mew=lw,label=lbl)
    else:
        ax.plot(dsort,cumProb,color=clr,ls=ls,lw=lw,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
xticks = list(plt.get(ax,'xticks'))+[notCompletedXtick]
xticklabels = xticks[:]
xticklabels[-1] = 'not\ncompleted'
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
xlim = [np.nanmin(passSession)-0.5,notCompletedXtick+0.5]
ax.set_xlim(xlim)
ax.set_ylim([0,1.02])
ax.set_xlabel('sessions to pass')
ax.set_ylabel('cum. prob.')
ax.legend(loc='lower right',fontsize=8)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for r,clr in enumerate('rgb'):
    ind = reg==r+1
    d = passSession[ind]
    d[np.isnan(d)] = notCompletedXtick
    dsort = np.sort(d)
    cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
    lbl = 'regimen '+str(r+1)+' (n='+str(d.size)+')'
    if len(cumProb)<2 or np.all(cumProb==1):
        ax.plot(dsort,cumProb,mrk,mec=clr,mfc='none',label=lbl)
    else:
        ax.plot(dsort,cumProb,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_ylim([0,1.02])
ax.set_xlabel('sessions to pass')
ax.set_ylabel('cum. prob.')
ax.legend(loc='lower right',fontsize=8)
plt.tight_layout()


stage = 'stage 5'
allReg = []
runningAllReg = []
timeoutsAllReg = []
longAllReg = []
passIndAllReg =[]
passBySession = []
handoffMice5 = []
handoffSessionStartTimes5 = []
for reg in (1,2,3,4):
    running = []
    timeouts = []
    long = []
    passInd = []
    stage5Mice = []
    dprimeCrossModal = []
    firstBlockVis = []
    for mid in mouseIds:
        if str(mid) in sheets:
            mouseInd = np.where(allMiceDf['mouse id']==mid)[0][0]
            if regimen[mouseInd]!=reg:
                continue
            if craniotomy[mouseInd]:
                pass#continue
            df = sheets[str(mid)]
            sessions = np.array([(stage in task and not 'templeton' in task) for task in df['task version']])
            nSessions = np.sum(sessions)
            if nSessions==0:
                continue
            allReg.append(reg)
            running.append(not allMiceDf.loc[mouseInd,'wheel fixed'])
            timeouts.append(allMiceDf.loc[mouseInd,'timeouts'])
            long.append(np.any(['long' in task for task in df['task version']]))
            passInd.append(np.nan)
            oriFirst = np.array(['ori tone' in task for task in df[sessions]['task version']])
            hits = np.array([[int(s) for s in re.findall('[0-9]+',d)] for d in df[sessions]['hits']])
            dprimeSame = np.array([[float(s) for s in re.findall('-*[0-9].[0-9]*',d)] for d in df[sessions]['d\' same modality']])
            dprimeOther = np.array([[float(s) for s in re.findall('-*[0-9].[0-9]*',d)] for d in df[sessions]['d\' other modality go stim']])
            stage5Mice.append(mid)
            dprimeCrossModal.append(dprimeOther)
            firstBlockVis.append(oriFirst)
            p = np.sum(np.all(np.stack((dprimeSame,dprimeOther))>1.5,axis=0),axis=1)>3
            pi = np.where(p[:-1] & p[1:])[0]
            if len(pi) > 0:
                passInd[-1] = pi[0]+1
                passBySession.append(np.full(50,np.nan))
                passBySession[-1][:p[passInd[-1]-1:].size] = p[passInd[-1]-1:]
                handoffMice5.append(str(mid))
                # handoffSessionStartTimes5.append(list(df['start time'][sessions][pi[0]:pi[0]+2]))
                handoffSessionStartTimes5.append(list(df['start time'][sessions][pi[0]:]))

    fig = plt.figure(figsize=(12,8))
    fig.suptitle('Stage 5 inter-modality d\'')
    nMice = len(dprimeCrossModal)
    for ind,(d,mid,vis,pi) in enumerate(zip(dprimeCrossModal,stage5Mice,firstBlockVis,passInd)):
        nSessions,nBlocks = d.shape
        ax = fig.add_subplot(1,nMice,ind+1)
        cmax = np.absolute(d).max()
        im = ax.imshow(d,cmap='bwr',clim=(-cmax,cmax))
        for i in range(nSessions):
            for j in range(nBlocks):
                ax.text(j,i,str(round(d[i,j],2)),ha='center',va='center',fontsize=6)
        ax.set_xticks(np.arange(nBlocks))
        ax.set_xticklabels(np.arange(nBlocks)+1)
        yticks = np.arange(nSessions) if nSessions<10 else np.concatenate(([0],np.arange(4,nSessions,5)))
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks+1)
        ax.set_ylim([nSessions-0.5,-0.5])
        ax.set_xlabel('block')
        if ind==0:
            ax.set_ylabel('session')
        for y,v in enumerate(vis):
            lbl = ''
            if v:
                lbl += 'vis first'
            if y==pi:
                if v:
                    lbl += ', '
                lbl += '*pass*'
            ax.text(nBlocks-0.4,y,lbl,ha='left',va='center',fontsize=8)
        ax.set_title(str(mid)+'\n'+'regimen '+str(reg),fontsize=10)
    plt.tight_layout()
    
    runningAllReg.extend(running)
    timeoutsAllReg.extend(timeouts)
    longAllReg.extend(long)
    passIndAllReg.extend(passInd)

reg,passInd,running,timeouts,long = [np.array(d) for d in (allReg,passIndAllReg,runningAllReg,timeoutsAllReg,longAllReg)]
passSession = passInd+1

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
d[np.isnan(d)] = notCompletedXtick
for ind,clr,ls,lw,mrk,lbl in zip((running,~running),
                      'kkk',('-','--'),(1,1),'oo',
                      ('run','no run')):
    d = passSession[ind]
    d[np.isnan(d)] = notCompletedXtick
    dsort = np.sort(d)
    cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
    lbl += ' (n='+str(d.size)+')'
    if len(cumProb)<2 or np.all(cumProb==1):
        ax.plot(dsort,cumProb,mrk,mec=clr,mfc='none',mew=lw,label=lbl)
    else:
        ax.plot(dsort,cumProb,color=clr,ls=ls,lw=lw,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
xticks = list(plt.get(ax,'xticks'))+[notCompletedXtick]
xticklabels = xticks[:]
xticklabels[-1] = 'not\ncompleted'
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
xlim = [np.nanmin(passSession)-0.5,notCompletedXtick+0.5]
ax.set_xlim(xlim)
ax.set_ylim([0,1.02])
ax.set_xlabel('sessions to pass')
ax.set_ylabel('cum. prob.')
ax.legend(loc='lower right',fontsize=8)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for r,clr in enumerate('rgbk'):
    ind = reg==r+1
    d = passSession[ind]
    d[np.isnan(d)] = notCompletedXtick
    dsort = np.sort(d)
    cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
    lbl = 'regimen '+str(r+1)+' (n='+str(d.size)+')'
    if len(cumProb)<2 or np.all(cumProb==1):
        ax.plot(dsort,cumProb,mrk,mec=clr,mfc='none',label=lbl)
    else:
        ax.plot(dsort,cumProb,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim(xlim)
ax.set_ylim([0,1.02])
ax.set_xlabel('sessions to pass')
ax.set_ylabel('cum. prob.')
ax.legend(loc='lower right',fontsize=8)
plt.tight_layout()
  
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
n = np.sum(~np.isnan(passBySession),axis=0)
m = np.nanmean(passBySession,axis=0)
s = np.nanstd(passBySession,axis=0)/(np.sum(~np.isnan(passBySession),axis=0)**0.5)
x = np.arange(m.size)+1
ax.plot(x,m,'k')
ax.fill_between(x,m+s,m-s,color='k',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,18])
ax.set_ylim([0,1.02])
ax.set_xlabel('session')
ax.set_ylabel('fraction above pass threshold')
plt.tight_layout()

handoffSessions = []
for mid,st in zip(handoffMice5,handoffSessionStartTimes5):
    for t in st:
        f = os.path.join(baseDir,'Data',mid,'DynamicRouting1_' + mid + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5')
        obj = DynRoutData()
        obj.loadBehavData(f)
        handoffSessions.append(obj)

   

# transition analysis
blockData = []
for obj in exps:
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        d = {'mouseID':obj.subjectName,
             'sessionStartTime': obj.startTime,
             'blockNum':blockInd+1,
             'goStim':goStim,
             'numAutoRewards':obj.autoRewarded[:10].sum()}
        blockTrials = obj.trialBlock == blockInd + 1
        for trials,lbl in zip((obj.goTrials,obj.sameModalNogoTrials,obj.otherModalGoTrials,obj.otherModalNogoTrials),
                              ('goTrials','sameModalNogoTrials','otherModalGoTrials','otherModalNogoTrials')):
            trials = trials & blockTrials
            d[lbl] = {'startTimes':obj.stimStartTimes[trials]-obj.blockFirstStimTimes[blockInd],
                      'response':obj.trialResponse[trials],
                      'responseTime':obj.responseTimes[trials]}
        blockData.append(d)
        

for blockType,rewModalColor,otherModalColor in zip(('vis','sound'),'gm','mg'):
    goLabel = 'visual' if blockType=='vis' else 'auditory'
    nogoLabel = 'auditory' if goLabel=='visual' else 'visual'
    blocks = [d for d in blockData if blockType in d['goStim']]
    nBlocks = len(blocks)
    nMice = len(set(d['mouseID'] for d in blockData))
    nSessions = len(set(d['sessionStartTime'] for d in blockData))
    nTrials = [len(d['goTrials']['response']) for d in blocks] + [len(d['otherModalGoTrials']['response']) for d in blocks]
    print('n trials: '+str(min(nTrials))+', '+str(max(nTrials))+', '+str(np.median(nTrials)))
    
    title = goLabel+' rewarded (' + str(nBlocks) +' blocks, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)'
    
    blockDur = 700
    binSize = 30
    bins = np.arange(0,blockDur+binSize/2,binSize)
    respRateGoTime = np.zeros((nBlocks,bins.size))
    respRateNogoTime = respRateGoTime.copy() 
    respRateOtherGoTime = respRateGoTime.copy()
    respRateOtherNogoTime = respRateGoTime.copy()
    latencyGoTime = respRateGoTime.copy()
    latencyNogoTime = respRateGoTime.copy()
    latencyOtherGoTime = respRateGoTime.copy()
    latencyOtherNogoTime = respRateGoTime.copy()
    
    hitTrials = np.zeros(nBlocks,dtype=int)
    falseAlarmTrials = hitTrials.copy()
    maxTrials = 100
    hitRateTrials = np.full((nBlocks,maxTrials),np.nan)
    falseAlarmRateTrials = hitRateTrials.copy()  
    hitLatencyTrials = hitRateTrials.copy()
    falseAlarmLatencyTrials = hitRateTrials.copy()
    
    for i,d in enumerate(blocks):
        for trials,r,lat in zip(('goTrials','sameModalNogoTrials','otherModalGoTrials','otherModalNogoTrials'),
                                (respRateGoTime,respRateNogoTime,respRateOtherGoTime,respRateOtherNogoTime),
                                (latencyGoTime,latencyNogoTime,latencyOtherGoTime,latencyOtherNogoTime)):
            c = np.zeros(bins.size)
            for trialInd,binInd in enumerate(np.digitize(d[trials]['startTimes'],bins)):
                r[i][binInd] += d[trials]['response'][trialInd]
                lat[i][binInd] += d[trials]['responseTime'][trialInd]
                c[binInd] += 1
            r[i] /= c
            lat[i] /= c
        for trials,n,r,lat in zip(('goTrials','otherModalGoTrials'),(hitTrials,falseAlarmTrials),(hitRateTrials,falseAlarmRateTrials),(hitLatencyTrials,falseAlarmLatencyTrials)):
            n[i] = d[trials]['response'].size
            r[i,:n[i]] = d[trials]['response']
            lat[i,:n[i]] = d[trials]['responseTime']
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    binTimes = bins+binSize/2
    for d,clr,ls,lbl in zip((respRateGoTime,respRateNogoTime,respRateOtherGoTime,respRateOtherNogoTime),
                            (rewModalColor,rewModalColor,otherModalColor,otherModalColor),('-','--','-','--'),
                            (goLabel+' go',goLabel+' nogo',nogoLabel+' go',nogoLabel+' nogo')):
        n = np.sum(~np.isnan(d),axis=0)
        m = np.nanmean(d,axis=0)
        m[n<10] = np.nan
        s = np.nanstd(d,axis=0)/(n**0.5)
        ax.plot(binTimes,m,clr,ls=ls,label=lbl+' stimulus')
        ax.fill_between(binTimes,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,700,100))
    ax.set_xlim([0,615])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Time (s); auto-rewards excluded')
    ax.set_ylabel('Response Rate')
    ax.legend(bbox_to_anchor=(1,0.5))
    ax.set_title(title)  
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for d,clr,lbl in zip((latencyGoTime,latencyOtherGoTime),(rewModalColor,otherModalColor),(goLabel,nogoLabel)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(binTimes,m,clr,label=lbl+' go')
        ax.fill_between(binTimes,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,700,100))
    ax.set_xlim([0,615])
    ax.set_yticks(np.arange(0.2,0.7,0.1))
    ax.set_ylim([0.25,0.65])
    ax.set_xlabel('Time (s); auto-rewards excluded')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(title='stimulus:',loc='lower right')
    ax.set_title(title)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    trialNum = np.arange(maxTrials)+1
    for d,clr,lbl in zip((hitRateTrials,falseAlarmRateTrials),(rewModalColor,otherModalColor),(goLabel,nogoLabel)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(trialNum,m,clr,label=lbl+' go')
        ax.fill_between(trialNum,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,25,5))
    ax.set_xlim([0,20])
    ax.set_ylim([0,1])
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Rate')
    ax.legend(title='stimulus:',loc='lower right')
    ax.set_title(title)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for d,clr,lbl in zip((hitLatencyTrials,falseAlarmLatencyTrials),(rewModalColor,otherModalColor),(goLabel,nogoLabel)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(trialNum,m,clr,label=lbl+' go')
        ax.fill_between(trialNum,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,25,5))
    ax.set_xlim([0,20])
    ax.set_yticks(np.arange(0.2,0.7,0.1))
    ax.set_ylim([0.25,0.65])
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(title='stimulus:',loc='lower right')
    ax.set_title(title)
    plt.tight_layout()


for blockType in ('visual','auditory'):
    goStim = 'vis' if blockType=='visual' else 'sound'
    nTransitions = 0    
    goProb = []
    goProbPrev = []
    goProbFirst = []
    nogoProb = []
    nogoProbPrev = []
    nogoProbFirst = []
    otherGoProb = []
    otherGoProbPrev = []
    otherGoProbFirst = []
    otherNogoProb = []
    otherNogoProbPrev = []
    otherNogoProbFirst = []
    goLat = []
    goLatPrev = []
    goLatFirst = []
    nogoLat = []
    nogoLatPrev = [] 
    nogoLatFirst = []
    otherGoLat = []
    otherGoLatPrev = []
    otherGoLatFirst = []
    otherNogoLat = []
    otherNogoLatPrev = []
    otherNogoLatFirst = []
    for i,block in enumerate(blockData):
        if goStim in block['goStim']:
            if block['blockNum'] > 1:
                nTransitions += 1
                prevBlock = blockData[i-1]
                goProb.append(block['goTrials']['response'])
                goProbPrev.append(prevBlock['otherModalGoTrials']['response'])
                nogoProb.append(block['sameModalNogoTrials']['response'])
                nogoProbPrev.append(prevBlock['otherModalNogoTrials']['response'])
                otherGoProb.append(block['otherModalGoTrials']['response'])
                otherGoProbPrev.append(prevBlock['goTrials']['response'])
                otherNogoProb.append(block['otherModalNogoTrials']['response'])
                otherNogoProbPrev.append(prevBlock['sameModalNogoTrials']['response'])
                
                goLat.append(block['goTrials']['responseTime'])
                goLatPrev.append(prevBlock['otherModalGoTrials']['responseTime'])
                nogoLat.append(block['sameModalNogoTrials']['responseTime'])
                nogoLatPrev.append(prevBlock['otherModalNogoTrials']['responseTime'])
                otherGoLat.append(block['otherModalGoTrials']['responseTime'])
                otherGoLatPrev.append(prevBlock['goTrials']['responseTime'])
                otherNogoLat.append(block['otherModalNogoTrials']['responseTime'])
                otherNogoLatPrev.append(prevBlock['sameModalNogoTrials']['responseTime'])
            else:
                goProbFirst.append(block['goTrials']['response'])
                nogoProbFirst.append(block['sameModalNogoTrials']['response'])
                otherGoProbFirst.append(block['otherModalGoTrials']['response'])
                otherNogoProbFirst.append(block['otherModalNogoTrials']['response'])
                
                goLatFirst.append(block['goTrials']['responseTime'])
                nogoLatFirst.append(block['sameModalNogoTrials']['responseTime'])
                otherGoLatFirst.append(block['otherModalGoTrials']['responseTime'])
                otherNogoLatFirst.append(block['otherModalNogoTrials']['responseTime'])
    
    nMice = len(set(d['mouseID'] for d in blockData))
    nSessions = len(set(d['sessionStartTime'] for d in blockData))
    title = (blockType+' rewarded blocks\n'
             'mean and 95% ci across transitions\n('+
             str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    colors,labels = ('gm',('visual','auditory')) if blockType=='visual' else ('mg',('auditory','visual'))
    
    preTrials = postTrials = 15 # 15, 45
    x = np.arange(-preTrials,postTrials+1)
    xlim =[-preTrials,postTrials]
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0,1.01]
    ax.plot([0,0],ylim,'k--')
    for first,clr,modal in zip(((goProbFirst,nogoProbFirst),(otherGoProbFirst,otherNogoProbFirst)),colors,labels):
        for r,ls,stim in zip(first,('-','--'),('go','nogo')):
            d = np.full((len(r),preTrials+postTrials+1),np.nan)
            for i,a in enumerate(r):
                j = min(postTrials,a.size)
                d[i,preTrials+1:preTrials+1+j] = a[:j]
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Probability')
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(blockType+' rewarded first block\n('+str(len(goProbFirst)) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0.25,0.6]
    ax.plot([0,0],ylim,'k--')
    for first,clr,modal in zip(((goLatFirst,nogoLatFirst),(otherGoLatFirst,otherNogoLatFirst)),colors,labels):
        for r,ls,stim in zip(first,('-','--'),('go','nogo')):
            d = np.full((len(r),preTrials+postTrials+1),np.nan)
            for i,a in enumerate(r):
                j = min(postTrials,a.size)
                d[i,preTrials+1:preTrials+1+j] = a[:j]
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(blockType+' rewarded first block\n('+str(len(goLatFirst)) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0,1.01]
    ax.plot([0,0],ylim,'k--')
    for a,clr,modal in zip((((goProb,goProbPrev),(nogoProb,nogoProbPrev)),((otherGoProb,otherGoProbPrev),(otherNogoProb,otherNogoProbPrev))),colors,labels):
        for b,ls,stim in zip(a,('-','--'),('go','nogo')):
            current,prev = b
            d = np.full((nTransitions,preTrials+postTrials+1),np.nan)
            for i,r in enumerate(prev):
                j = len(r) if len(r)<preTrials else preTrials
                d[i,preTrials-j:preTrials] = r[-j:] 
            for i,r in enumerate(current):
                j = len(r) if len(r)<postTrials else postTrials
                d[i,preTrials+1:preTrials+1+j] = r[:j] 
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Probability')
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title('transitions to '+blockType+' rewarded blocks\n('+str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ylim = [0.25,0.6]
    ax.plot([0,0],ylim,'k--')
    for a,clr,modal in zip((((goLat,goLatPrev),(nogoLat,nogoLatPrev)),((otherGoLat,otherGoLatPrev),(otherNogoLat,otherNogoLatPrev))),colors,labels):
        for b,ls,stim in zip(a,('-','--'),('go','nogo')):
            current,prev = b
            d = np.full((nTransitions,preTrials+postTrials+1),np.nan)
            for i,r in enumerate(prev):
                j = len(r) if len(r)<preTrials else preTrials
                d[i,preTrials-j:preTrials] = r[-j:] 
            for i,r in enumerate(current):
                j = len(r) if len(r)<postTrials else postTrials
                d[i,preTrials+1:preTrials+1+j] = r[:j] 
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
            ax.plot(x,m,clr,ls=ls,label=modal+' '+stim+' stimulus')
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Latency (s)')
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title('transitions to '+blockType+' rewarded blocks\n('+str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for blockType,goStim,clr in zip(('visual','auditory'),(('vis1','sound1'),('sound1','vis1')),'gm'):
    for stim,trials,ls,lbl in zip(goStim,('goTrials','otherModalGoTrials'),('-',':'),('rewarded','unrewarded')):
        d = [block[trials]['responseTime'] for block in blockData if block['goStim']==stim]
        d = np.concatenate(d)
        d = d[~np.isnan(d)]
        dsort = np.sort(d)
        cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,ls=ls,label=blockType+' '+lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0.1,1])
ax.set_ylim([0,1.01])
ax.set_xlabel('Response Latency (s)')
ax.set_ylabel('Cumulative Probability')
ax.legend(loc='lower right')
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xmax = 9
for goStim,clr,lbl in zip(('vis1','sound1'),'mg',('auditory','visual')):
    trialsToNogo = []
    firstNogoResp = []
    for block in blockData:
        if block['goStim']==goStim:
            trialsToNogo.append(np.sum(block['goTrials']['startTimes'] < block['otherModalGoTrials']['startTimes'][0]) + block['numAutoRewards'])
            firstNogoResp.append(block['otherModalGoTrials']['response'][0])
    trialsToNogo = np.array(trialsToNogo)
    firstNogoResp = np.array(firstNogoResp)
    x = np.unique(trialsToNogo)
    r = [firstNogoResp[trialsToNogo==n] for n in x]
    n = [len(d) for d in r]
    m = np.array([np.nanmean(d) for d in r])
    ci = [np.percentile([np.nanmean(np.random.choice(d,len(d),replace=True)) for _ in range(5000)],(2.5,97.5)) for d in r]
    ax.plot(x[x<=xmax],m[x<=xmax],color=clr,label=lbl)
    for i,c in enumerate(ci):
        if i<=xmax:
            ax.plot([x[i]]*2,c,clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.5,xmax+0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Preceeding rewarded modality (go stim) trials, including autorewards')
ax.set_ylabel('Probability of responding\nto first non-rewarded modality (go stim) trial')
ax.legend()
plt.tight_layout()


trialsSincePrevResp = {blockType: {trialType: [] for trialType in ('goTrials','otherModalGoTrials')} for blockType in ('visual','auditory')}
firstTrialInBlockResp = copy.deepcopy(trialsSincePrevResp)
trialsSincePrevRespInBlock = copy.deepcopy(trialsSincePrevResp)
respInBlock = copy.deepcopy(trialsSincePrevResp)
for obj in exps:
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        blockType,otherModalGoStim = ('visual','sound1') if 'vis' in goStim else ('auditory','vis1')
        blockTrials = obj.trialBlock==blockInd+1
        for trials,trialType in zip((obj.goTrials,obj.otherModalGoTrials),('goTrials','otherModalGoTrials')):
            trials = trials & blockTrials
            trialInd = np.where(trials)[0]
            trialResp = obj.trialResponse[trials]
            for i in range(1,len(trialInd)):
                if trialResp[i-1]:
                    t = trialInd[i]-trialInd[i-1]
                    trialsSincePrevRespInBlock[blockType][trialType].append(t)
                    respInBlock[blockType][trialType].append(trialResp[i])
            if blockInd>0:
                stimTrials = obj.trialStim==goStim if trialType=='goTrials' else obj.trialStim==otherModalGoStim
                stimInd = np.where(stimTrials & ~obj.autoRewarded)[0]
                prevTrial = stimInd[np.where(stimInd==trialInd[0])[0][0]-1]
                if obj.trialResponse[prevTrial]:
                    t = trialInd[0]-prevTrial
                    trialsSincePrevResp[blockType][trialType].append(t)
                    firstTrialInBlockResp[blockType][trialType].append(obj.trialResponse[trialInd[0]])
            
            
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for blockType,mec in zip(('visual','auditory'),'gm'):
    for trialType in ('goTrials','otherModalGoTrials'):
        clr = 'g' if (blockType=='visual' and trialType=='goTrials') or (blockType=='auditory' and trialType=='otherModalGoTrials') else 'm'
        mfc = clr if trialType=='goTrials' else 'none'
        for n in np.unique(trialsSincePrevRespInBlock[blockType][trialType]):
            r = np.array(respInBlock[blockType][trialType])[trialsSincePrevRespInBlock[blockType][trialType]==n]
            m = r.mean()
            ci = np.percentile([np.nanmean(np.random.choice(r,r.size,replace=True)) for _ in range(100)],(2.5,97.5))
            ax.plot(n,m,'o',mec=clr,mfc=mfc)
            ax.plot([n,n],ci,color=clr)
            
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for blockType,mec in zip(('visual','auditory'),'gm'):
    for trialType,mfc in zip(('otherModalGoTrials',),('none',)):
        clr = 'g' if (blockType=='visual' and trialType=='goTrials') or (blockType=='auditory' and trialType=='otherModalGoTrials') else 'm'
        mfc = clr if trialType=='goTrials' else 'none'
        for n in np.unique(trialsSincePrevResp[blockType][trialType]):
            r = np.array(firstTrialInBlockResp[blockType][trialType])[trialsSincePrevResp[blockType][trialType]==n]
            m = r.mean()
            ci = np.percentile([np.nanmean(np.random.choice(r,r.size,replace=True)) for _ in range(100)],(2.5,97.5))
            ax.plot(n,m,'o',mec=clr,mfc=mfc)
            ax.plot([n,n],ci,color=clr)
            



# running
exps = []
for mid,st,run in zip(handoffMice5,handoffSessionStartTimes5,running[~np.isnan(passInd)]):
    if run:
        for t in st:
            f = os.path.join(baseDir,'Data',mid,'DynamicRouting1_' + mid + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5')
            obj = DynRoutData()
            obj.loadBehavData(f)
            exps.append(obj)

nMice = len(set(obj.subjectName for obj in exps))


fig = plt.figure(figsize=(8,8))
fig.suptitle(str(len(exps))+' sessions, '+str(nMice)+' mice')
gs = matplotlib.gridspec.GridSpec(3,2)
axs = []
ymax = 0
preTime = 1.5
postTime = 3
runPlotTime = np.arange(-preTime,postTime+1/obj.frameRate,1/obj.frameRate)
for stim in ('vis1','vis2','sound1','sound2','catch',None):
    if stim is None:
        i = 2
        j = 1
    elif stim=='catch':
        i = 2
        j = 0
    else:
        i = 0 if '1' in stim else 1
        j = 0 if 'vis' in stim else 1
    ax = fig.add_subplot(gs[i,j])
    axs.append(ax)
    for blockRew,clr in zip(('vis','sound'),'gm'):
        speed = []
        if stim is not None:
            for obj in exps:
                stimTrials = (obj.trialStim==stim) & (~obj.autoRewarded)
                blockTrials = np.array([blockRew in s for s in obj.rewardedStim])
                for st in obj.stimStartTimes[stimTrials & blockTrials]:
                    if st >= preTime and st+postTime <= obj.frameTimes[-1]:
                        ind = (obj.frameTimes >= st-preTime) & (obj.frameTimes <= st+postTime)
                        speed.append(np.interp(runPlotTime,obj.frameTimes[ind]-st,obj.runningSpeed[ind]))
        if len(speed) > 0:
            m = np.nanmean(speed,axis=0)
            s = np.nanstd(speed,axis=0)/(len(speed)**0.5)
        else:
            m = s = np.full(runPlotTime.size,np.nan)
        ax.plot(runPlotTime,m,color=clr,label=blockRew+' rewarded')
        ax.fill_between(runPlotTime,m+s,m-s,color=clr,alpha=0.25)
        ymax = max(ymax,np.nanmax(m+s))
    if stim is None:
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='center')
    else:
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0,20])
        if i==2 and j==0:
            ax.set_xlabel('time from stimulus onset (s)')
        if i==1 and j==0:
            ax.set_ylabel('running speed (cm/s)')
        ax.set_title(stim)
for ax in axs:
    ax.set_ylim([0,1.05*ymax])
plt.tight_layout()


visSpeed = []
soundSpeed = []
for rewStim,speed in zip(('vis1','sound1'),(visSpeed,soundSpeed)):
    for obj in exps:
        speed.append(np.mean([np.nanmean(obj.runningSpeed[sf-obj.quiescentFrames:sf]) for sf in obj.stimStartFrame[obj.rewardedStim==rewStim]]))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
alim = [0,1.05*max(visSpeed+soundSpeed)]
ax.plot(alim,alim,'--',color='0.5')
ax.plot(visSpeed,soundSpeed,'ko')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('run speed, vis rewarded (cm/s)')
ax.set_ylabel('run speed, sound rewarded (cm/s)')
ax.set_title(str(len(exps))+' sessions, '+str(nMice)+' mice')
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,100],[0,100],'--',color='0.5')
amax = 0
for rew,mfc in zip((True,False),('k','none')):
    respSpeed = []
    noRespSpeed = []
    for resp,speed in zip((True,False),(respSpeed,noRespSpeed)):
        for obj in exps:
            respTrials = obj.trialResponse if resp else ~obj.trialResponse
            s = []
            for stim in ('vis1','sound1'):
                stimTrials = obj.trialStim==stim
                blockTrials = obj.rewardedStim==stim if rew else obj.rewardedStim!=stim
                s.append([np.nanmean(obj.runningSpeed[sf-obj.quiescentFrames:sf]) for sf in obj.stimStartFrame[respTrials & stimTrials & blockTrials]])
            speed.append(np.mean(np.concatenate(s)))
    lbl = 'stim rewarded' if rew else 'stim not rewarded'
    ax.plot(respSpeed,noRespSpeed,'o',mec='k',mfc=mfc,label=lbl)
    amax = max(amax,1.05*max(respSpeed+noRespSpeed))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,amax])
ax.set_ylim([0,amax])
ax.set_aspect('equal')
ax.set_xlabel('run speed, response (cm/s)')
ax.set_ylabel('run speed, no response (cm/s)')
ax.legend()
ax.set_title(str(len(exps))+' sessions, '+str(nMice)+' mice')
plt.tight_layout()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for tol,clr in zip((1,5),'rb'):
    for matchedResp,ls in zip((False,True),('-','--')):
        lbl = str(tol)+' cm/s tolerance'
        if matchedResp:
            lbl += ', response matched'
        matchedTrials = []
        for obj in exps:
            speed = np.array([np.nanmean(obj.runningSpeed[sf-obj.quiescentFrames:sf]) for sf in obj.stimStartFrame])
            r = (obj.trialResponse,~obj.trialResponse) if matchedResp else (np.ones(obj.nTrials,dtype=bool),)*2
            for respTrials in r:
                for stim in ('vis1','sound1'):
                    stimTrials = (obj.trialStim==stim) & (~obj.autoRewarded)
                    vs,ss = [speed[stimTrials & respTrials & np.array([blockRew in s for s in obj.rewardedStim])] for blockRew in ('vis','sound')]
                    for v in vs:
                        if v>0:
                            matchedTrials.append(np.sum((ss>v-tol) & (ss<v+tol)))
                    for s in ss:
                        if s>0:
                            matchedTrials.append(np.sum((vs>s-tol) & (vs<s+tol)))
        d = np.array(matchedTrials)
        dsort = np.sort(d)
        cumProb = [np.sum(d<=i)/d.size for i in dsort]
        ax.plot(dsort,cumProb,color=clr,ls=ls,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1.01])
ax.set_xlabel('number of matched trials')
ax.set_ylabel('cumulative probability')
ax.legend()
plt.tight_layout()                



# timeouts
stageNum = []
regimenNum = []
timeoutDur = []
falseAlarms = []
for mid in mouseIds:
    if str(mid) in sheets:
        mouseInd = np.where(allMiceDf['mouse id']==mid)[0][0]
        if craniotomy[mouseInd]:
            continue
        mid = str(mid)
        mouseDir = os.path.join(baseDir,'Data',mid)
        if not os.path.isdir(mouseDir):
            continue
        behavFiles = glob.glob(os.path.join(mouseDir,'*.hdf5'))
        exps = []
        for f in behavFiles:
            obj = DynRoutData()
            obj.loadBehavData(f)
            exps.append(obj)
        exps = sortExps(exps)
        for obj in exps:
            stage = re.findall('stage ([0-9])',obj.taskVersion)
            if len(stage)>0:
                stageNum.append(int(stage[0]))
                regimenNum.append(regimen[mouseInd])
                timeoutDur.append(obj.incorrectTimeoutFrames/obj.frameRate)
                falseAlarms.append(obj.falseAlarmTrials.sum())
stageNum = np.array(stageNum)
regimenNum = np.array(regimenNum)
timeoutDur = np.array(timeoutDur)
falseAlarms = np.array(falseAlarms)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for stage,reg,clr in zip((1,2,3,3,4),((1,2),(1,2),1,2,(1,2)),'rgbcm'):
    ind = (stageNum==stage) & np.in1d(regimenNum,reg) & (timeoutDur>0)
    d = falseAlarms[ind] * 3 / 3600
    dsort = np.sort(d)
    cumProb = [np.sum(d<=i)/d.size for i in dsort]
    lbl = 'stage '+str(stage)
    if isinstance(reg,int):
        lbl += ', regimen '+str(reg)
    ax.plot(dsort,cumProb,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1.02])
ax.set_xlabel('fraction of session in timeout')
ax.set_ylabel('fraction of sessions')
ax.legend()
plt.tight_layout()



# contrast, volume
for obj in exps:
    stimNames = ('vis1','vis2','autorewarded','sound1','sound2','catch')
    preTime = 4
    postTime = 4
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        fig = plt.figure(figsize=(8,8))
        fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim)
        gs = matplotlib.gridspec.GridSpec(3,2)
        blockTrials = obj.trialBlock == blockInd + 1
        for stimInd,stim in enumerate(stimNames):
            if stim=='autorewarded':
                trials = obj.autoRewarded
            elif stim=='catch':
                trials = obj.catchTrials
            else:
                trials = (obj.trialStim==stim) & (~obj.autoRewarded)
            trials = trials & blockTrials
            i,j = (stimInd,0) if stimInd<3 else (stimInd-3,1)
            ax = fig.add_subplot(gs[i,j])
            ax.add_patch(matplotlib.patches.Rectangle([-obj.quiescentFrames/obj.frameRate,0],width=obj.quiescentFrames/obj.frameRate,height=trials.sum()+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
            ax.add_patch(matplotlib.patches.Rectangle([obj.responseWindowTime[0],0],width=np.diff(obj.responseWindowTime),height=trials.sum()+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
            for i,st in enumerate(obj.stimStartTimes[trials]):
                lt = obj.lickTimes - st
                trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
                ax.vlines(trialLickTimes,i+0.5,i+1.5,colors='k')       
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([-preTime,postTime])
            ax.set_ylim([0.5,trials.sum()+0.5])
            ax.set_yticks([1,trials.sum()])
            ax.set_xlabel('time from stimulus onset (s)')
            ax.set_ylabel('trial')
            title = stim + ', reponse rate=' + str(round(obj.trialResponse[trials].sum()/trials.sum(),2))
            ax.set_title(title)   
        fig.tight_layout()

for obj in exps:    
    stimNames = ('vis1','vis2','sound1','sound2')
    fig = plt.figure(figsize=(8,6))
    gs = matplotlib.gridspec.GridSpec(2,2)
    for stimInd,stim in enumerate(stimNames):
        i,j = (stimInd,0) if stimInd<2 else (stimInd-2,1)
        ax = fig.add_subplot(gs[i,j])
        stimTrials = ((obj.trialStim==stim) | obj.catchTrials) & (~obj.autoRewarded)
        trialLevel,xlbl = (obj.trialVisContrast,'contrast') if 'vis' in stim else (obj.trialSoundVolume,'volume')
        for blockInd,(goStim,txty) in enumerate(zip(obj.blockStimRewarded,(1.03,1.1))):
            blockTrials = obj.trialBlock == blockInd + 1
            trials = blockTrials & stimTrials
            levels = np.unique(trialLevel)
            r = []
            n = []
            for s in np.unique(levels):
                tr = trials & (trialLevel == s)
                n.append(tr.sum())
                r.append(obj.trialResponse[tr].sum() / tr.sum())
            clr = 'g' if 'vis' in goStim else 'm'
            ax.plot(levels,r,'o',color=clr,label='block '+str(blockInd+1)+', '+goStim+' rewarded')
            for x,txt in zip(levels,n):
                ax.text(x,txty,str(txt),ha='center',va='bottom',fontsize=8)    
            try:
                bounds = ((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf))
                fitParams = fitCurve(calcWeibullDistrib,levels,r,bounds=bounds)
            except:
                fitParams = None
            if fitParams is not None:
                fitX = np.arange(0,max(levels)+0.0001,0.0001)
                ax.plot(fitX,calcWeibullDistrib(fitX,*fitParams),clr)  
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim([0,1.03])
        ax.set_xlabel(xlbl)
        ax.set_ylabel('response rate')
        ax.set_title(stim,y=1.12)
        if i==0 and j==0:
            ax.legend(loc='upper left',fontsize=8)
    plt.tight_layout()

    

# multimodal stimuli
for obj in exps:
    stimNames = ('vis1','vis2','vis1+sound1','vis1+sound2','autorewarded',
                 'sound1','sound2','vis2+sound1','vis2+sound2','catch')
    preTime = 4
    postTime = 4
    respTime = []
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        fig = plt.figure(figsize=(8,8))
        fig.suptitle('block ' + str(blockInd+1) + ': go=' + goStim)
        gs = matplotlib.gridspec.GridSpec(5,2)
        blockTrials = obj.trialBlock == blockInd + 1
        respTime.append([])
        for stimInd,stim in enumerate(stimNames):
            if stim=='autorewarded':
                trials = obj.autoRewarded
            elif stim=='catch':
                trials = obj.catchTrials
            else:
                trials = (obj.trialStim==stim) & (~obj.autoRewarded)
            trials = trials & blockTrials
            respTime[-1].append(obj.responseTimes[trials & obj.trialResponse])
            i,j = (stimInd,0) if stimInd<5 else (stimInd-5,1)
            ax = fig.add_subplot(gs[i,j])
            ax.add_patch(matplotlib.patches.Rectangle([-obj.quiescentFrames/obj.frameRate,0],width=obj.quiescentFrames/obj.frameRate,height=trials.sum()+1,facecolor='r',edgecolor=None,alpha=0.2,zorder=0))
            ax.add_patch(matplotlib.patches.Rectangle([obj.responseWindowTime[0],0],width=np.diff(obj.responseWindowTime),height=trials.sum()+1,facecolor='g',edgecolor=None,alpha=0.2,zorder=0))
            for i,st in enumerate(obj.stimStartTimes[trials]):
                lt = obj.lickTimes - st
                trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
                ax.vlines(trialLickTimes,i+0.5,i+1.5,colors='k')       
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([-preTime,postTime])
            ax.set_ylim([0.5,trials.sum()+0.5])
            ax.set_yticks([1,trials.sum()])
            ax.set_xlabel('time from stimulus onset (s)')
            ax.set_ylabel('trial')
            title = stim + ', reponse rate=' + str(round(obj.trialResponse[trials].sum()/trials.sum(),2))
            ax.set_title(title)   
        fig.tight_layout()
    
    fig = plt.figure(figsize=(8,8))
    gs = matplotlib.gridspec.GridSpec(5,2)
    for stimInd,stim in enumerate(stimNames):
        i,j = (stimInd,0) if stimInd<5 else (stimInd-5,1)
        ax = fig.add_subplot(gs[i,j])
        for blockInd,(blockRt,goStim) in enumerate(zip(respTime,obj.blockStimRewarded)):
            rt = blockRt[stimInd]
            rtSort = np.sort(rt)
            cumProb = [np.sum(rt<=i)/rt.size for i in rtSort]
            clr = 'g' if 'vis' in goStim else 'm'
            ax.plot(rtSort,cumProb,color=clr,label='block '+str(blockInd+1)+', '+goStim+' rewarded')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,obj.responseWindowTime[1]])
        ax.set_ylim([0,1.02])
        ax.set_xlabel('response time (s)')
        ax.set_ylabel('cum. prob.')
        ax.set_title(stim)
        if i==0 and j==1:
            ax.legend(loc='lower right',fontsize=8)
    plt.tight_layout()



# learning summary plots
hitRate = []
falseAlarmRate = []
falseAlarmSameModal = []
falseAlarmOtherModalGo = []
falseAlarmOtherModalNogo = []
catchRate = []
blockReward = []
for obj in exps:
    hitRate.append(obj.hitRate)
    falseAlarmRate.append(obj.falseAlarmRate)
    falseAlarmSameModal.append(obj.falseAlarmSameModal)
    falseAlarmOtherModalGo.append(obj.falseAlarmOtherModalGo)
    falseAlarmOtherModalNogo.append(obj.falseAlarmOtherModalNogo)
    catchRate.append(obj.catchResponseRate)
    blockReward.append(obj.blockStimRewarded)
hitRate = np.array(hitRate)
falseAlarmRate = np.array(falseAlarmRate)
falseAlarmSameModal = np.array(falseAlarmSameModal)
falseAlarmOtherModalGo = np.array(falseAlarmOtherModalGo)
falseAlarmOtherModalNogo = np.array(falseAlarmOtherModalNogo)
catchRate = np.array(catchRate)    

fig = plt.figure(figsize=(12,8))
nBlocks = hitRate.shape[1]
nExps = len(exps)
if nExps>40:
    yticks = np.arange(0,nExps,10)
elif nExps > 10:
    yticks = np.arange(0,nExps,5)
else:
    yticks = np.arange(nExps)
for ind,(r,lbl) in enumerate(zip((hitRate,falseAlarmSameModal,falseAlarmOtherModalGo,falseAlarmOtherModalNogo,catchRate),
                               ('hit rate','false alarm Same','false alarm diff go','false alarm diff nogo','catch rate'))):  
    ax = fig.add_subplot(1,5,ind+1)
    im = ax.imshow(r,cmap='magma',clim=(0,1))
    for i in range(nExps):
        for j in range(nBlocks):
            ax.text(j,i,str(round(r[i,j],2)),ha='center',va='center',fontsize=6)
    ax.set_xticks(np.arange(nBlocks))
    ax.set_xticklabels(np.arange(nBlocks)+1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks+1)
    ax.set_ylim([nExps-0.5,-0.5])
    ax.set_xlabel('block')
    if ind==0:
        ax.set_ylabel('session')
        cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
        cb.set_ticks([0,0.5,1])
    if ind==4:
        for y,rew in enumerate(blockReward):
            ax.text(nBlocks,y,list(rew)[:2],ha='left',va='center',fontsize=8)
    ax.set_title(lbl)
plt.tight_layout()

    

 
# learning summary plots (old)
hitRate = []
falseAlarmRate = []
falseAlarmSameModal = []
falseAlarmOtherModalGo = []
falseAlarmOtherModalNogo = []
catchRate = []
blockReward = []
for obj in exps:
    if ((obj.taskVersion in ('vis sound vis detect','sound vis sound detect','vis sound detect','sound vis detect')
        and len(obj.blockStimRewarded)>=3) or
        ('vis sound discrim' in obj.taskVersion or 'sound vis discrim' in obj.taskVersion) or
        ('ori tone discrim' in obj.taskVersion or 'tone ori discrim' in obj.taskVersion) or
        ('ori sweep discrim' in obj.taskVersion or 'sweep ori discrim' in obj.taskVersion)):
        hitRate.append(obj.hitRate)
        falseAlarmRate.append(obj.falseAlarmRate)
        falseAlarmSameModal.append(obj.falseAlarmSameModal)
        falseAlarmOtherModalGo.append(obj.falseAlarmOtherModalGo)
        falseAlarmOtherModalNogo.append(obj.falseAlarmOtherModalNogo)
        catchRate.append(obj.catchResponseRate)
        blockReward.append(obj.blockStimRewarded)
hitRate = np.array(hitRate)
falseAlarmRate = np.array(falseAlarmRate)
falseAlarmSameModal = np.array(falseAlarmSameModal)
falseAlarmOtherModalGo = np.array(falseAlarmOtherModalGo)
falseAlarmOtherModalNogo = np.array(falseAlarmOtherModalNogo)
catchRate = np.array(catchRate)    

fig = plt.figure(figsize=(10,5))
nBlocks = hitRate.shape[1]
nExps = len(exps)
if nExps>40:
    yticks = np.arange(0,nExps,10)
elif nExps > 10:
    yticks = np.arange(0,nExps,5)
else:
    yticks = np.arange(nExps)
for i,(r,lbl) in enumerate(zip((hitRate,falseAlarmSameModal,falseAlarmOtherModalGo,falseAlarmOtherModalNogo,catchRate),
                               ('hit rate','false alarm Same','false alarm diff go','false alarm diff nogo','catch rate'))):  
    ax = fig.add_subplot(1,5,i+1)
    im = ax.imshow(r,cmap='magma',clim=(0,1))
    ax.set_xticks(np.arange(nBlocks))
    ax.set_xticklabels(np.arange(nBlocks)+1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks+1)
    ax.set_ylim([nExps-0.5,-0.5])
    ax.set_xlabel('block')
    if i==0:
        ax.set_ylabel('session')
        cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
        cb.set_ticks([0,0.5,1])
    if i==4:
        for y,rew in enumerate(blockReward):
            ax.text(nBlocks,y,list(rew)[:2],ha='left',va='center',fontsize=8)
    ax.set_title(lbl)
plt.tight_layout()


hitRate = []
falseAlarmRate = []
catchRate = []
blockReward = []
for obj in exps:
    if 'ori discrim' in obj.taskVersion or 'sound discrim' in obj.taskVersion or 'tone discrim' in obj.taskVersion or 'sweep discrim' in obj.taskVersion:
        hitRate.append(obj.hitRate)
        falseAlarmRate.append(obj.falseAlarmRate)
        catchRate.append(obj.catchResponseRate)
        blockReward.append(obj.blockStimRewarded)
hitRate = np.array(hitRate).squeeze()
falseAlarmRate = np.array(falseAlarmRate).squeeze()
catchRate = np.array(catchRate).squeeze()

fig = plt.figure(figsize=(6,9))
nExps = len(exps)
if nExps>40:
    yticks = np.arange(0,nExps,10)
elif nExps > 10:
    yticks = np.arange(0,nExps,5)
else:
    yticks = np.arange(nExps)
ax = fig.add_subplot(1,1,1)
im = ax.imshow(np.stack((hitRate,falseAlarmRate,catchRate),axis=1),cmap='magma',clim=(0,1))
ax.set_xticks([0,1,2])
ax.set_xticklabels(('hit','false alarm','catch'),rotation=90)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks+1)
ax.set_ylim([nExps-0.5,-0.5])
ax.set_ylabel('session')
cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.15)
cb.set_ticks([0,0.5,1])
for y,rew in enumerate(blockReward):
    ax.text(3,y,list(rew),ha='left',va='center',fontsize=8)
plt.tight_layout()


# for shawn
fig = plt.figure(figsize=(6,9))
ax = fig.add_subplot(1,1,1)
im = ax.imshow(np.stack((hitRate,falseAlarmRate,catchRate),axis=1)[:59],cmap='magma',clim=(0,1))
ax.set_xticks([0,1,2])
ax.set_xticklabels(('hit','false alarm','catch'),rotation=90)
ax.set_ylabel('session')
cb = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.15)
cb.set_ticks([0,0.5,1])
cb.set_label('response rate')
rprev = ''
for y,rew in enumerate(blockReward[:59]):
    r = list(rew[0])
    if rprev != '' and r != rprev:
        ax.text(3,y,'switch',ha='left',va='center',fontsize=8)
    rprev = r
plt.tight_layout()

fig = plt.figure(figsize=(11,3.5))
ax = fig.add_subplot(1,1,1)
x = np.arange(59)+1
xticks = np.arange(0,60,10)
rprev = ''
for i,rew in enumerate(blockReward[:59]):
    r = rew[0]
    if rprev != '' and r != rprev:
        ax.plot([x[i]]*2,[0,1],'k--')
        ax.text(x[i],1.025,'switch',ha='center',va='baseline')
    rprev = r
for y,clr,lbl in zip((catchRate,falseAlarmRate,hitRate),('0.8','m','g'),('catch','nogo','go')):
    ax.plot(x,y[:59],'o-',ms=4,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xlim([0,60])
ax.set_ylim([0,1.01])
ax.set_xlabel('Session')
ax.set_ylabel('Response Rate')
ax.legend(loc='upper left')
plt.tight_layout()



# ori
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rr = []
for obj,clr in zip(exps,plt.cm.tab20(np.linspace(0,1,len(exps)))):
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        blockTrials = (obj.trialBlock == blockInd + 1) & ~obj.autoRewarded & ~obj.catchTrials
        oris = np.unique(obj.trialGratingOri)
        r = []
        for ori in oris:
            trials = blockTrials & (obj.trialGratingOri == ori)
            r.append(obj.trialResponse[trials].sum() / trials.sum())
        ax.plot(oris,r,'o-',color=clr,alpha=0.5)
        rr.append(r)
mean = np.mean(rr,axis=0)
sem = np.std(rr,axis=0)/(len(exps)**0.5)
ax.plot(oris,mean,'ko-',ms=8,lw=2,label=lbl)
for x,m,s in zip(oris,mean,sem):
    ax.plot([x,x],[m-s,m+s],'k-',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1.02])
ax.set_xlabel('ori (0=go, >0=nogo)')
ax.set_ylabel('response rate')
plt.tight_layout()
                
                
# contrast
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
hr = []
far = []
for obj,clr in zip(exps,plt.cm.tab20(np.linspace(0,1,len(exps)))):
    for blockInd,goStim in enumerate(obj.blockStimRewarded):
        blockTrials = obj.trialBlock == blockInd + 1
        for trials,lbl in zip((obj.goTrials,obj.nogoTrials),('go','nogo')):
            r = []
            for c in obj.visContrast:
                tr = trials & blockTrials & (obj.trialVisContrast == c)
                r.append(obj.trialResponse[tr].sum() / tr.sum())
            ls,mfc = ('-',clr) if lbl=='go' else ('--','none')
            ax.plot(obj.visContrast,r,'o',color=clr,ls=ls,mec=clr,mfc=mfc,alpha=0.5)
            if lbl=='go':
                hr.append(r)
            else:
                far.append(r)
for r,lbl in zip((hr,far),('go','nogo')):
    mean = np.mean(r,axis=0)
    sem = np.std(r,axis=0)/(len(exps)**0.5)
    ls,mfc = ('-','k') if lbl=='go' else ('--','none')
    ax.plot(obj.visContrast,mean,'ko-',mfc=mfc,ls=ls,ms=8,lw=2,label=lbl)
    for x,m,s in zip(obj.visContrast,mean,sem):
        ax.plot([x,x],[m-s,m+s],'k-',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xscale('log')
ax.set_ylim([0,1.02])
ax.set_xlabel('contrast')
ax.set_ylabel('response rate')
ax.legend()
plt.tight_layout()
    
    

# sound latency test
filePath = fileIO.getFile(rootDir=os.path.join(baseDir,'Data'),fileType='*.hdf5')

d = h5py.File(filePath,'r')
    
frameRate = 60
frameIntervals = d['frameIntervals'][:]
frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))

trialEndFrame = d['trialEndFrame'][:]
nTrials = trialEndFrame.size
trialStartFrame = d['trialStartFrame'][:nTrials]
stimStartFrame = d['trialStimStartFrame'][:nTrials]
stimStartTimes = frameTimes[stimStartFrame]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
mic = d['microphoneData'][:]
frame = np.arange(-30,45)
for sf in stimStartFrame:
    ax.plot(frame,mic[sf-30:sf+45],'k')
    
d.close()


