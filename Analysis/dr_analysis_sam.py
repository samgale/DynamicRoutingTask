# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:55:44 2021

@author: svc_ccg
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
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
    passInd = []
    reg1PassInd = []
    fig,axs = plt.subplots(2)
    fig.set_size_inches(8,6)
    xmax = 0
    for mid in mouseIds:
        if str(mid) in sheets:
            mouseInd = np.where(allMiceDf['mouse id']==mid)[0][0]
            if craniotomy[mouseInd]:
                continue
            running.append(not allMiceDf.loc[mouseInd,'wheel fixed'])
            timeouts.append(allMiceDf.loc[mouseInd,'timeouts'])
            df = sheets[str(mid)]
            sessions = np.array([(stage in task and not 'templeton' in task) for task in df['task version']])
            nSessions = np.sum(sessions)
            hits = np.array([int(re.findall('[0-9]+',s)[0]) for s in df[sessions]['hits']])
            dprime = np.array([float(re.findall('-*[0-9].[0-9]*',s)[0]) for s in df[sessions]['d\' same modality']])
            passInd.append(np.nan)
            reg1PassInd.append(np.nan)
            for i in range(nSessions):
                if i > 0:
                    if all(hits[i-1:i+1] > 100) and all(dprime[i-1:i+1] > 1.5):
                        passInd[-1] = i
                        break
            else:
                if any(str(int(stage[-1])+1) in task for task in df['task version']):
                    passInd[-1] = nSessions-1
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
            xmax = max(xmax,nSessions+0.5)
            ls = '-' if running[-1] else '--'
            clr = 'm' if timeouts[-1] else 'g'
            lbl = 'run' if running[-1] else 'no run'
            lbl += ', timeouts' if timeouts[-1] else ', no timeouts'
            for ax,val in zip(axs,(hits,dprime)):
                if np.isnan(passInd[-1]):
                    ax.plot(x,val,color=clr,ls=ls,label=lbl)
                else:
                    ax.plot(x[:passInd[-1]+1],val[:passInd[-1]+1],color=clr,ls=ls,label=lbl)
                    ax.plot(passInd[-1]+1,val[passInd[-1]],'o',mec=clr,mfc='none')
    for i,(ax,ylbl,thresh) in enumerate(zip(axs,('hits','d prime'),(100,1.5))):
        ax.plot([0,xmax],[thresh]*2,'k:',zorder=0)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        xticks = np.arange(0,xmax+1,5) if stage=='stage 1' else np.arange(xmax)
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
    
    passInd,reg1PassInd,running,timeouts = [np.array(d) for d in (passInd,reg1PassInd,running,timeouts)]
    passSession = passInd+1
    reg1PassSession = reg1PassInd+1
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for r1,r2,run,to in zip(reg1PassSession,passSession,running,timeouts):
        if not np.isnan(r1):
            ls = '-' if run else '--'
            clr = 'm' if to else 'g'
            ax.plot([0,1],[r1,r2],'o-',color=clr,mfc='none',ls=ls)
    ax.plot([0,1],[np.nanmedian(reg1PassSession),np.nanmedian(passSession)],'ko-',ms=10)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['>150 hits\n(regimen 1)','>100 hits\n(regimen 2)'])
    ax.set_xlim([-0.25,1.25])
    ax.set_ylim([1,max(np.nanmax(reg1PassSession),np.nanmax(passSession))+1])
    ax.set_ylabel('sessions to pass')
    ax.set_title(stage+' (regimen 1 mice)')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(3,1,1)
    for d,ls,lbl in zip((passSession[running],passSession[~running]),('-','--'),('run','no run')):
        d = d[~np.isnan(d)]
        dsort = np.sort(d)
        cumProb = [np.sum(d<=i)/d.size for i in dsort]
        lbl += ' (n='+str(d.size)+')'
        ax.plot(dsort,cumProb,color='k',ls=ls,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    xlim = [np.nanmin(passSession)-0.5,np.nanmax(passSession)+0.5]
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.02])
    ax.set_ylabel('cum. prob.')
    ax.legend(loc='lower right',fontsize=8)
    ax.set_title(stage)
    
    ax = fig.add_subplot(3,1,2)
    for d,clr,lbl in zip((passSession[timeouts],passSession[~timeouts]),'mg',('timeouts','no timeouts')):
        d = d[~np.isnan(d)]
        dsort = np.sort(d)
        cumProb = [np.sum(d<=i)/d.size for i in dsort]
        lbl += ' (n='+str(d.size)+')'
        ax.plot(dsort,cumProb,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.02])
    ax.set_ylabel('cum. prob.')
    ax.legend(loc='lower right',fontsize=8)
    
    ax = fig.add_subplot(3,1,3)
    for ind,clr,ls,lbl in zip((running & timeouts,running & ~timeouts,~running & timeouts,~running & ~timeouts),
                          'mgmg',('-','-','--','--'),('run, timeouts','run, no timeouts','no run, timeouts','no run, no timeouts')):
        d = passSession[ind]
        d = d[~np.isnan(d)]
        dsort = np.sort(d)
        cumProb = [np.sum(d<=i)/d.size for i in dsort]
        lbl += ' (n='+str(d.size)+')'
        ax.plot(dsort,cumProb,color=clr,ls=ls,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.02])
    ax.set_xlabel('sessions to pass')
    ax.set_ylabel('cum. prob.')
    ax.legend(loc='lower right',fontsize=8)
    plt.tight_layout()
    
    
stage = 'stage 3'
for reg,hitThresh,substage in zip((1,2,2),(150,50,50),(1,1,2)):
    running = []
    timeouts = []
    passInd = []
    fig,axs = plt.subplots(3)
    fig.set_size_inches(8,6)
    xmax = 0
    for mid in mouseIds:
        if str(mid) in sheets:
            mouseInd = np.where(allMiceDf['mouse id']==mid)[0][0]
            if regimen[mouseInd]!=reg or craniotomy[mouseInd]:
                continue
            running.append(not allMiceDf.loc[mouseInd,'wheel fixed'])
            timeouts.append(allMiceDf.loc[mouseInd,'timeouts'])
            df = sheets[str(mid)]
            sessions = np.array([(stage in task and not 'templeton' in task) for task in df['task version']])
            if substage==2:
                ind = np.where(['stage 3 tone' in task for task in df['task version']])[0]
                if len(ind)==0:
                    continue
                else:
                    sessions[:ind[0]] = False
            nextStage = 'stage 4' if reg==1 or substage==2 else 'stage 3 tone'
            nextStageInd = np.where([nextStage in task for task in df['task version']])[0]
            if len(nextStageInd)>0:
                sessions[nextStageInd[0]:] = False
            nSessions = np.sum(sessions)
            if nSessions==0:
                continue
            hits = np.array([int(re.findall('[0-9]+',s)[0]) for s in df[sessions]['hits']])
            dprimeSame = np.array([float(re.findall('-*[0-9].[0-9]*',s)[0]) for s in df[sessions]['d\' same modality']])
            if reg==2:
                dprimeOther = np.array([float(re.findall('-*[0-9].[0-9]*',s)[0]) for s in df[sessions]['d\' other modality go stim']])
            else:
                dprimeOther = None
            passInd.append(np.nan)
            for i in range(nSessions):
                if i > 0 and all(hits[i-1:i+1] > hitThresh) and all(dprimeSame[i-1:i+1] > 1.5) and (reg==1 or all(dprimeOther[i-1:i+1] > 1.5)):
                    passInd[-1] = i
                    if regimen[mouseInd]==1:
                        break
            x = np.arange(nSessions)+1
            xmax = max(xmax,nSessions+0.5)
            ls = '-' if running[-1] else '--'
            clr = 'm' if timeouts[-1] else 'g'
            lbl = 'run' if running[-1] else 'no run'
            lbl += ', timeouts' if timeouts[-1] else ', no timeouts'
            for ax,val in zip(axs,(hits,dprimeSame,dprimeOther)):
                if val is not None:
                    ax.plot(x,val,color=clr,ls=ls,label=lbl)
                    if not np.isnan(passInd[-1]):
                        ax.plot(passInd[-1]+1,val[passInd[-1]],'o',mec=clr,mfc='none')
    for i,(ax,ylbl,thresh) in enumerate(zip(axs,('hits','d prime same','d prime other'),(hitThresh,1.5,1.5))):
        ax.plot([0,xmax],[thresh]*2,'k:',zorder=0)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(xmax+1))
        ax.set_xlim([0.5,xmax])
        ylim = ax.get_ylim()
        ax.set_ylim([min(0,ylim[0]),ylim[1]])
        if (reg==1 and i==1) or (reg==2 and i==2):
            ax.set_xlabel('session')
        ax.set_ylabel(ylbl)
        if i==0:
            handles,labels = ax.get_legend_handles_labels()
            lblDict = dict(zip(labels,handles))
            ax.legend(lblDict.values(),lblDict.keys(),loc='lower right',fontsize=8)
            title = stage+', regimen '+str(reg)
            if reg==2:
                title += ', part '+str(substage)
            ax.set_title(title)
    if reg==1:
        fig.delaxes(axs[2])
    plt.tight_layout()


stage = 'stage 4'
for version in ('blocks','modality'):
    running = []
    timeouts = []
    passInd = []
    stage4Mice = []
    dprimeCrossModal = []
    firstBlockVis = []
    fig,axs = plt.subplots(3,2)
    fig.set_size_inches(8,6)
    fig.suptitle(stage)
    xmax = 0
    for mid in mouseIds:
        if str(mid) in sheets:
            mouseInd = np.where(allMiceDf['mouse id']==mid)[0][0]
            if craniotomy[mouseInd]:
                continue
            df = sheets[str(mid)]
            sessions = np.array([(stage in task and not 'templeton' in task) for task in df['task version']])
            nSessions = np.sum(sessions)
            if nSessions==0:
                continue
            running.append(not allMiceDf.loc[mouseInd,'wheel fixed'])
            timeouts.append(allMiceDf.loc[mouseInd,'timeouts'])
            oriFirst = np.array(['ori tone' in task for task in df[sessions]['task version']])
            hits = np.array([[int(s) for s in re.findall('[0-9]+',d)] for d in df[sessions]['hits']])
            dprimeSame = np.array([[float(s) for s in re.findall('-*[0-9].[0-9]*',d)] for d in df[sessions]['d\' same modality']])
            dprimeOther = np.array([[float(s) for s in re.findall('-*[0-9].[0-9]*',d)] for d in df[sessions]['d\' other modality go stim']])
            stage4Mice.append(mid)
            dprimeCrossModal.append(dprimeOther)
            firstBlockVis.append(oriFirst)
            passInd.append(np.nan)
            for i in range(nSessions):
                if i > 0 and np.all(dprimeSame[i-1:i+1] > 1.5) and np.all(dprimeOther[i-1:i+1] > 1.5):
                    passInd[-1] = i
                    if regimen[mouseInd]==1:
                        break
            x = np.arange(nSessions)+1
            xmax = max(xmax,nSessions+0.5)
            ls = '-' if running[-1] else '--'
            clr = 'm' if timeouts[-1] else 'g'
            lbl = 'run' if running[-1] else 'no run'
            lbl += ', timeouts' if timeouts[-1] else ', no timeouts'
            for i,val in enumerate((hits,dprimeSame,dprimeOther)):
                for j in (0,1):
                    if version=='blocks':
                        v = val[:,j]
                    elif j==0:
                        v = val[np.stack((oriFirst,~oriFirst),axis=-1)]
                    else:
                        v = val[np.stack((~oriFirst,oriFirst),axis=-1)]
                    axs[i,j].plot(x,v,color=clr,ls=ls,label=lbl)
                    if not np.isnan(passInd[-1]):
                        axs[i,j].plot(passInd[-1]+1,v[passInd[-1]],'o',mec=clr,mfc='none')
    for i,ylbl in enumerate(('hits','dprime same','dprime other')):
        for j in (0,1):
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
                    title = 'block 1' if version=='blocks' else 'vis block'
                    ax.set_title(title)
                    handles,labels = ax.get_legend_handles_labels()
                    lblDict = dict(zip(labels,handles))
                    ax.legend(lblDict.values(),lblDict.keys(),loc='lower right',fontsize=8)
                else:
                    title = 'block 2' if version=='blocks' else 'sound block'
                    ax.set_title(title)
    plt.tight_layout()

fig = plt.figure(figsize=(12,8))
fig.suptitle('Stage 4 cross-modal d\'')
nMice = len(dprimeCrossModal)
for ind,(d,mid,vis,pi) in enumerate(zip(dprimeCrossModal,stage4Mice,firstBlockVis,passInd)):
    if not np.isnan(pi):
        d = d[:pi+1]
        vis = vis[:pi+1]
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
    ax.set_title(mid)
plt.tight_layout()


stage = 'stage 5'
running = []
timeouts = []
passInd = []
stage5Mice = []
dprimeCrossModal = []
firstBlockVis = []
for mid in mouseIds:
    if str(mid) in sheets:
        mouseInd = np.where(allMiceDf['mouse id']==mid)[0][0]
        if craniotomy[mouseInd]:
            continue
        df = sheets[str(mid)]
        sessions = np.array([(stage in task and not 'templeton' in task) for task in df['task version']])
        nSessions = np.sum(sessions)
        if nSessions==0:
            continue
        running.append(not allMiceDf.loc[mouseInd,'wheel fixed'])
        timeouts.append(allMiceDf.loc[mouseInd,'timeouts'])
        oriFirst = np.array(['ori tone' in task for task in df[sessions]['task version']])
        hits = np.array([[int(s) for s in re.findall('[0-9]+',d)] for d in df[sessions]['hits']])
        dprimeSame = np.array([[float(s) for s in re.findall('-*[0-9].[0-9]*',d)] for d in df[sessions]['d\' same modality']])
        dprimeOther = np.array([[float(s) for s in re.findall('-*[0-9].[0-9]*',d)] for d in df[sessions]['d\' other modality go stim']])
        stage5Mice.append(mid)
        dprimeCrossModal.append(dprimeOther)
        firstBlockVis.append(oriFirst)
        passInd.append(np.nan)
        for i in range(nSessions):
            if i > 0 and np.all(dprimeSame[i-1:i+1] > 1.5) and np.all(dprimeOther[i-1:i+1] > 1.5):
                passInd[-1] = i
                if regimen[mouseInd]==1:
                    break

fig = plt.figure(figsize=(12,8))
fig.suptitle('Stage 5 cross-modal d\'')
nMice = len(dprimeCrossModal)
for ind,(d,mid,vis,pi) in enumerate(zip(dprimeCrossModal,stage5Mice,firstBlockVis,passInd)):
    if not np.isnan(pi):
        d = d[:pi+1]
        vis = vis[:pi+1]
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
    ax.set_title(mid)
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
        for trials,lbl in zip((obj.goTrials,obj.otherModalGoTrials),('goTrials','nogoTrials')):
            trials = trials & blockTrials
            d[lbl] = {'startTimes':obj.stimStartTimes[trials]-obj.blockFirstStimTimes[blockInd],
                      'response':obj.trialResponse[trials],
                      'responseTime':obj.responseTimes[trials]}
        blockData.append(d)
        
[(d['mouseID'],d['sessionStartTime'],d['numAutoRewards']) for d in blockData]

for blockType,hitColor,faColor in zip(('vis','sound'),'gm','mg'):
    goLabel = 'vis' if blockType=='vis' else 'aud'
    nogoLabel = 'aud' if goLabel=='vis' else 'vis'
    blocks = [d for d in blockData if blockType in d['goStim']]
    nBlocks = len(blocks)
    nMice = len(set(d['mouseID'] for d in blockData))
    nSessions = len(set(d['sessionStartTime'] for d in blockData))
    nTrials = [len(d['goTrials']['response']) for d in blocks] + [len(d['nogoTrials']['response']) for d in blocks]
    print('n trials: '+str(min(nTrials))+', '+str(max(nTrials))+', '+str(np.median(nTrials)))
    
    title = goLabel+' rewarded (' + str(nBlocks) +' blocks, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)'
    
    blockDur = 700
    binSize = 30
    bins = np.arange(0,blockDur+binSize/2,binSize)
    hitRateTime = np.zeros((nBlocks,bins.size))
    falseAlarmRateTime = hitRateTime.copy()  
    hitLatencyTime = hitRateTime.copy()
    falseAlarmLatencyTime = hitRateTime.copy()
    
    hitTrials = np.zeros(nBlocks,dtype=int)
    falseAlarmTrials = hitTrials.copy()
    maxTrials = 100
    hitRateTrials = np.full((nBlocks,maxTrials),np.nan)
    falseAlarmRateTrials = hitRateTrials.copy()  
    hitLatencyTrials = hitRateTrials.copy()
    falseAlarmLatencyTrials = hitRateTrials.copy()
    
    for i,d in enumerate(blocks):
        for trials,r,lat in zip(('goTrials','nogoTrials'),(hitRateTime,falseAlarmRateTime),(hitLatencyTime,falseAlarmLatencyTime)):
            c = np.zeros(bins.size)
            for trialInd,binInd in enumerate(np.digitize(d[trials]['startTimes'],bins)):
                r[i][binInd] += d[trials]['response'][trialInd]
                lat[i][binInd] += d[trials]['responseTime'][trialInd]
                c[binInd] += 1
            r[i] /= c
            lat[i] /= c
        for trials,n,r,lat in zip(('goTrials','nogoTrials'),(hitTrials,falseAlarmTrials),(hitRateTrials,falseAlarmRateTrials),(hitLatencyTrials,falseAlarmLatencyTrials)):
            n[i] = d[trials]['response'].size
            r[i,:n[i]] = d[trials]['response']
            lat[i,:n[i]] = d[trials]['responseTime']
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    binTimes = bins+binSize/2
    for d,clr,lbl in zip((hitRateTime,falseAlarmRateTime),(hitColor,faColor),(goLabel,nogoLabel)):
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(binTimes,m,clr,label=lbl+' go')
        ax.fill_between(binTimes,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,700,100))
    ax.set_xlim([0,615])
    ax.set_ylim([0,1])
    ax.set_xlabel('Time (s); auto-rewards excluded')
    ax.set_ylabel('Response Rate')
    ax.legend(title='stimulus:',loc='lower right')
    ax.set_title(title)  
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for d,clr,lbl in zip((hitLatencyTime,falseAlarmLatencyTime),(hitColor,faColor),(goLabel,nogoLabel)):
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
    for d,clr,lbl in zip((hitRateTrials,falseAlarmRateTrials),(hitColor,faColor),(goLabel,nogoLabel)):
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
    for d,clr,lbl in zip((hitLatencyTrials,falseAlarmLatencyTrials),(hitColor,faColor),(goLabel,nogoLabel)):
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
    nogoProb = []
    nogoProbPrev = []
    goLat = []
    goLatPrev = []
    nogoLat = []
    nogoLatPrev = [] 
    for block in blockData:
        if goStim in block['goStim'] and block['blockNum'] > 1:
            nTransitions += 1
            for d in blockData:
                if d['mouseID']==block['mouseID'] and d['blockNum']==block['blockNum']-1:
                    prevBlock = d
                    break
            goProb.append(block['goTrials']['response'])
            goProbPrev.append(prevBlock['nogoTrials']['response'])
            nogoProb.append(block['nogoTrials']['response'])
            nogoProbPrev.append(prevBlock['goTrials']['response'])
            goLat.append(block['goTrials']['responseTime'])
            goLatPrev.append(prevBlock['nogoTrials']['responseTime'])
            nogoLat.append(block['nogoTrials']['responseTime'])
            nogoLatPrev.append(prevBlock['goTrials']['responseTime'])
    
    title = (blockType+' rewarded blocks\n'
             'mean and 95% ci across transitions\n('+
             str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    colors,labels = ('gm',('visual','auditory')) if blockType=='visual' else ('mg',('auditory','visual'))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(21)
    for prev,current,clr,lbl in zip((goProbPrev,nogoProbPrev),(goProb,nogoProb),colors,labels):
        d = np.full((nTransitions,21),np.nan)
        d[:,0] = [r[-1] for r in prev]
        for i,r in enumerate(current):
            j = len(r) if len(r)<20 else 20
            d[i,1:j+1] = r[:j] 
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(np.sum(~np.isnan(d),axis=0)**0.5)
        ax.plot(-1,m[0],'o',color=clr)
        ax.plot([-1,-1],[m[0]-s[0],m[0]+s[0]],clr)
        ax.plot(x[1:],m[1:],clr,label=lbl+' go stimulus')
        ax.fill_between(x[1:],(m+s)[1:],(m-s)[1:],color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([-1,1,5,10,15,20])
    ax.set_xlim([-2,15])
    ax.set_ylim([0,1])
    ax.set_xlabel('Trial Number (of indicated type, excluding auto-rewards)')
    ax.set_ylabel('Response Probability')
    ax.legend(loc='lower right')
    ax.set_title(blockType+' rewarded blocks\n('+str(nTransitions) +' transitions, ' + str(nSessions) + ' sessions, ' + str(nMice)+' mice)')
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for prev,first,last,clr,lbl in zip((goProbPrev,nogoProbPrev),(goProb,nogoProb),(goProb,nogoProb),colors,labels):
        prev,first,last = [[r[i] for r in d] for d,i in zip((prev,first,last),(-1,0,-1))]
        m = [np.nanmean(d) for d in (prev,first,last)]
        ci = [np.percentile([np.nanmean(np.random.choice(d,len(d),replace=True)) for _ in range(5000)],(2.5,97.5)) for d in (prev,first,last)]
        ax.plot([0,1,2],m,'o-',color=clr,label=lbl+' go stimulus')
        for i,c in enumerate(ci):
            ax.plot([i,i],c,clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(('last trial of\nprevious block',
                        'first trial\nof block\nafter auto-rewards',
                        'last trial\nof block'))
    ax.set_xlim([-0.2,2.2])
    ax.set_ylim([0,1])
    ax.set_ylabel('Response Probability')
    ax.legend(loc='lower right')
    ax.set_title(title)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for prev,first,last,clr,lbl in zip((goLatPrev,nogoLatPrev),(goLat,nogoLat),(goLat,nogoLat),colors,labels):
        prev,first,last = [[r[i] for r in d] for d,i in zip((prev,first,last),(-1,0,-1))]
        m = [np.nanmean(d) for d in (prev,first,last)]
        ci = [np.percentile([np.nanmean(np.random.choice(d,len(d),replace=True)) for _ in range(5000)],(2.5,97.5)) for d in (prev,first,last)]
        ax.plot([0,1,2],m,'o-',color=clr,label=lbl+' stimulus (current block)')
        for i,c in enumerate(ci):
            ax.plot([i,i],c,clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(('last trial of\nprevious block',
                        'first trial\nof block\nafter auto-rewards',
                        'last trial\nof block'))
    ax.set_xlim([-0.2,2.2])
    ax.set_ylim([0.3,0.55])
    ax.set_ylabel('Response Latency (s)')
    ax.legend(loc='lower right')
    ax.set_title(title)
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


