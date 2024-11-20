import copy
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getSessionData


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

optoExps = pd.read_excel(os.path.join(baseDir,'Sam','OptoExperiments.xlsx'),sheet_name=None)


epoch = 'stim' # stim or feedback
hemi = 'bilateral' # unilateral, bilateral, or multilateral
hitThresh = 10
if epoch == 'feedback':
    areaNames = ('RSC','pACC','aACC','plFC','mFC','lFC')
    areaLabels = (('RSC',),('pACC',),('aACC',),('plFC',),('mFC',),('lFC',))
    dprime = {area: [] for area in areaNames}
    hitCount = copy.deepcopy(dprime)
else:
    if hemi == 'multilateral':
        areaNames = ('V1','V1','V1','lFC','lFC','lFC')
        areaLabels = (('V1',),('V1 left',),('V1 right',),('lFC',),('lFC left',),('lFC right',))
    else:
        areaNames = ('V1','PPC','RSC','pACC','aACC','plFC','mFC','lFC')
        areaLabels = (('V1','V1 left'),('PPC',),('RSC',),('pACC',),('aACC','ACC'),('plFC',),('mFC',),('lFC','PFC'))
    stimNames = ('vis1','vis2','sound1','sound2','catch')
    respRate = {area: {goStim: {opto: [] for opto in ('no opto','opto')} for goStim in ('vis1','sound1')} for area in areaNames}
    respTime = copy.deepcopy(respRate)
for mid in optoExps:
    df = optoExps[mid]
    sessions = [epoch in task for task in df['task version']]
    if hemi == 'unilateral':
        sessions = sessions & df['unilateral'] & ~df['bilateral']
    elif hemi == 'bilateral':
        sessions = sessions & ~df['unilateral'] & df['bilateral']
    elif hemi == 'multilateral':
        sessions = sessions & df['unilateral'] & df['bilateral']
    sessions = sessions & np.any(np.stack([df[area] for area in areaNames],axis=1),axis=1)
    if np.any(sessions):
        sessionData = [getSessionData(mid,startTime) for startTime in df['start time'][sessions]]
        for area,lbl in zip(areaNames,areaLabels):
            exps = [exp for exp,hasArea in zip(sessionData,df[area][sessions]) if hasArea]
            if len(exps) > 0:
                if epoch == 'stim':
                    for i,goStim in enumerate(('vis1','sound1')):
                        for optoLbl in ('no opto',lbl):
                            r = np.zeros(len(stimNames))
                            n = r.copy()
                            rt = r.copy()
                            rtn = r.copy()
                            for obj in exps:
                                blockTrials = (obj.rewardedStim==goStim) & (~obj.autoRewardScheduled) & (np.array(obj.hitCount)[obj.trialBlock-1] >= hitThresh)
                                optoTrials = obj.trialOptoLabel=='no opto' if optoLbl=='no opto' else np.in1d(obj.trialOptoLabel,lbl)
                                for j,stim in enumerate(stimNames):
                                    stimTrials = obj.trialStim == stim
                                    trials = blockTrials & optoTrials & stimTrials
                                    r[j] += obj.trialResponse[trials].sum()
                                    n[j] += trials.sum()
                                    zrt = (obj.responseTimes-np.nanmean(obj.responseTimes[stimTrials]))/np.nanstd(obj.responseTimes[stimTrials])
                                    rt[j] += np.nansum(zrt[trials])
                                    rtn[j] += np.sum(~np.isnan(zrt[trials]))
                            respRate[area][goStim]['no opto' if optoLbl=='no opto' else 'opto'].append(r/n)
                            respTime[area][goStim]['no opto' if optoLbl=='no opto' else 'opto'].append(rt/rtn)
                elif epoch == 'feedback':
                    dprime[area].append(np.mean([obj.dprimeOtherModalGo for obj in exps],axis=0))
                    hitCount[area].append(np.mean([obj.hitCount for obj in exps],axis=0))


# opto stim plots
xticks = np.arange(len(stimNames))
for area in areaNames:
    fig = plt.figure(figsize=(6,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        for opto,clr in zip(('no opto','opto'),'kb'):
            rr = respRate[area][goStim][opto]
            if i==0 and opto=='no opto':
                fig.suptitle(area + ' (n = ' + str(len(rr)) + ' mice)')
            if len(rr) > 0:
                mean = np.mean(rr,axis=0)
                sem = np.std(rr,axis=0)/(len(rr)**0.5)
                ax.plot(xticks,mean,color=clr,lw=2,label=opto)
                for x,m,s in zip(xticks,mean,sem):
                    ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        ax.set_xticks(xticks)
        if i==1:
            ax.set_xticklabels(stimNames)
        else:
            ax.set_xticklabels([])
        ax.set_xlim([-0.25,len(stimNames)-0.75])
        ax.set_ylim([-0.01,1.01])
        ax.set_ylabel('Response Rate',fontsize=14)
        ax.set_title(goStim + ' rewarded',fontsize=14)
        if i==0:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
    plt.tight_layout()
    

xticks = np.arange(2)
for area in areaNames:
    fig = plt.figure(figsize=(4,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        for opto,clr in zip(('no opto','opto'),'kb'):
            rr = respTime[area][goStim][opto]
            if i==0 and opto=='no opto':
                fig.suptitle(area + ' (n = ' + str(len(rr)) + ' mice)')
            if len(rr) > 0:
                mean = np.nanmean(rr,axis=0)[[0,2]]
                sem = np.nanstd(rr,axis=0)[[0,2]]/(len(rr)**0.5)
                ax.plot(xticks,mean,color=clr,lw=2,label=opto)
                for x,m,s in zip(xticks,mean,sem):
                    ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(xticks)
        if i==1:
            ax.set_xticklabels(('vis1','sound1'))
        else:
            ax.set_xticklabels([])
        ax.set_xlim([-0.25,len(xticks)-0.75])
        # ax.set_ylim([-0.01,1.01])
        ax.set_ylabel('Response Time (z score)')
        ax.set_title(goStim + ' rewarded')
        if i==0:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()
    

xticks = np.arange(len(stimNames))
for area in areaNames:
    fig = plt.figure(figsize=(5,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        ax.plot([-1,6],[0,0],'k--')
        rr = [np.array(respRate[area][goStim][opto]) for opto in ('no opto','opto')]
        rr = rr[1] - rr[0]
        if i==0:
            fig.suptitle(area + ' (n = ' + str(len(rr)) + ' mice)')
        if len(rr) > 0:
            for r in rr:
                ax.plot(xticks,r,color='k',lw=1,alpha=0.2)
            mean = np.mean(rr,axis=0)
            sem = np.std(rr,axis=0)/(len(rr)**0.5)
            ax.plot(xticks,mean,color='k',lw=2)
            for x,m,s in zip(xticks,mean,sem):
                ax.plot([x,x],[m-s,m+s],color='k',lw=2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(xticks)
        if i==1:
            ax.set_xticklabels(stimNames)
        else:
            ax.set_xticklabels([])
        ax.set_xlim([-0.25,len(xticks)-0.75])
        ax.set_ylim([-1,1])
        ax.set_ylabel(r'$\Delta$ Response Rate')
        ax.set_title(goStim + ' rewarded')
    plt.tight_layout()


xticks = np.arange(2)
for area in areaNames:
    fig = plt.figure(figsize=(4,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        ax.plot([-1,6],[0,0],'k--')
        rr = [np.array(respRate[area][goStim][opto]) for opto in ('no opto','opto')]
        if len(rr[0]) > 0:
            rr = 100 * ((rr[1][:,[0,2]] - rr[0][:,[0,2]]) / rr[0][:,[0,2]])
            if i==0:
                fig.suptitle(area + ' (n = ' + str(len(rr)) + ' mice)')
            for r in rr:
                ax.plot(xticks,r,color='k',lw=1,alpha=0.2)
            mean = np.mean(rr,axis=0)
            sem = np.std(rr,axis=0)/(len(rr)**0.5)
            ax.plot(xticks,mean,color='k',lw=2)
            for x,m,s in zip(xticks,mean,sem):
                ax.plot([x,x],[m-s,m+s],color='k',lw=2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(xticks)
        if i==1:
            ax.set_xticklabels(['vis1','sound1'])
        else:
            ax.set_xticklabels([])
        ax.set_xlim([-0.25,len(xticks)-0.75])
        ax.set_ylim([-100,100])
        ax.set_ylabel(r'$\Delta$ Response Rate (%)')
        ax.set_title(goStim + ' rewarded')
    plt.tight_layout()


# opto feedback plots
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(6) + 1
for area in areaNames:
    m = np.mean(dprime[area],axis=0)
    # m = np.mean(hitCount[area],axis=0)
    ax.plot(x,m,'k')




















