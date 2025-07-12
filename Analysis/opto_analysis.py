import copy
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getSessionData,getFirstExperimentSession


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

optoExps = pd.read_excel(os.path.join(baseDir,'Sam','OptoExperiments.xlsx'),sheet_name=None)

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)


genotype = 'VGAT-ChR2' # VGAT-ChR2 or wt control
epoch = 'feedback' # stim or feedback
hemi = 'bilateral' # unilateral, bilateral, or multilateral
hitThresh = 10

if epoch == 'stim':
    if hemi == 'multilateral':
        areaNames = ('V1','V1','V1','lFC','lFC','lFC')
        areaExperimentLabels = (('V1',),('V1 left',),('V1 right',),('lFC',),('lFC left',),('lFC right',))
        areaLabels = ('V1','V1 left','V1 right','lFC','lFC left','lFC right')
    else:
        areaNames = ('V1','PPC','RSC','pACC','aACC','plFC','mFC','lFC')
        areaExperimentLabels = (('V1','V1 left'),('PPC',),('RSC',),('pACC',),('aACC','ACC'),('plFC',),('mFC',),('lFC','PFC'))
        areaLabels = areaNames
    stimNames = ('vis1','vis2','sound1','sound2','catch')
    respRate = {lbl: {goStim: {opto: [] for opto in ('no opto','opto')} for goStim in ('vis1','sound1')} for lbl in areaLabels}
    respTime = copy.deepcopy(respRate)
    respRateRepeat = {lbl: {opto: [] for opto in ('no opto','opto')} for lbl in areaLabels}
    respRateNonRepeat = copy.deepcopy(respRateRepeat)
    respTimeRepeat = copy.deepcopy(respRateRepeat)
    respTimeNonRepeat = copy.deepcopy(respRateRepeat)
elif epoch == 'feedback':
    areaNames = ('RSC','pACC','aACC','plFC','mFC','lFC')
    areaExperimentLabels = (('RSC',),('pACC',),('aACC',),('plFC',),('mFC',),('lFC',))
    areaLabels = areaNames
    dprime = {lbl: [] for lbl in areaLabels+('control',)}
    hitCount = copy.deepcopy(dprime)
for mid in optoExps:
    df = optoExps[mid]
    if df['genotype'][0] != genotype:
        continue
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
        for area,expLbl,lbl in zip(areaNames,areaExperimentLabels,areaLabels):
            exps = [exp for exp,hasArea in zip(sessionData,df[area][sessions]) if hasArea]
            if len(exps) > 0:
                if epoch == 'stim':
                    for optoLbl in ('no opto',expLbl):
                        optoKey = 'no opto' if optoLbl=='no opto' else 'opto'
                        rRepeat = 0
                        nRepeat = 0
                        rNonRepeat = 0
                        nNonRepeat = 0
                        rtRepeat = 0
                        rtnRepeat = 0
                        rtNonRepeat = 0
                        rtnNonRepeat = 0
                        for i,goStim in enumerate(('vis1','sound1')):
                            r = np.zeros(len(stimNames))
                            n = r.copy()
                            rt = r.copy()
                            rtn = r.copy()
                            for obj in exps:
                                blockTrials = (obj.rewardedStim==goStim) & (~obj.autoRewardScheduled) & (np.array(obj.hitCount)[obj.trialBlock-1] >= hitThresh)
                                optoTrials = obj.trialOptoLabel=='no opto' if optoLbl=='no opto' else np.in1d(obj.trialOptoLabel,expLbl)
                                for j,stim in enumerate(stimNames):
                                    stimTrials = obj.trialStim == stim
                                    trials = blockTrials & optoTrials & stimTrials
                                    r[j] += obj.trialResponse[trials].sum()
                                    n[j] += trials.sum()
                                    rtz = (obj.responseTimes-np.nanmean(obj.responseTimes[stimTrials]))#/np.nanstd(obj.responseTimes[stimTrials])
                                    rt[j] += np.nansum(rtz[trials])
                                    rtn[j] += np.sum(~np.isnan(rtz[trials]))
                                    if stim == ('vis1' if goStim=='sound1' else 'sound1'):
                                        prevResp = obj.trialResponse[stimTrials][np.searchsorted(np.where(stimTrials)[0],np.where(trials)[0]) - 1]
                                        rRepeat += obj.trialResponse[trials][prevResp].sum()
                                        nRepeat += np.sum(prevResp)
                                        rNonRepeat += obj.trialResponse[trials][~prevResp].sum()
                                        nNonRepeat += np.sum(~prevResp)
                                        rtRepeat += np.nansum(rtz[trials][prevResp])
                                        rtnRepeat += np.sum(~np.isnan(rtz[trials][prevResp]))
                                        rtNonRepeat += np.nansum(rtz[trials][~prevResp])
                                        rtnNonRepeat += np.sum(~np.isnan(rtz[trials][~prevResp]))
                            respRate[lbl][goStim][optoKey].append(r/n)
                            respTime[lbl][goStim][optoKey].append(rt/rtn)
                        respRateRepeat[lbl][optoKey].append(rRepeat/nRepeat)
                        respRateNonRepeat[lbl][optoKey].append(rNonRepeat/nNonRepeat)
                        respTimeRepeat[lbl][optoKey].append(rtRepeat/rtnRepeat)
                        respTimeNonRepeat[lbl][optoKey].append(rtNonRepeat/rtnNonRepeat)
                elif epoch == 'feedback':
                    dprime[lbl].append(np.mean([obj.dprimeOtherModalGo for obj in exps],axis=0))
                    hitCount[lbl].append(np.mean([obj.hitCount for obj in exps],axis=0))
        if epoch == 'feedback':
            df = drSheets[mid] if mid in drSheets else nsbSheets[mid]
            firstExp = getFirstExperimentSession(df)
            controlSessions = [getSessionData(mid,startTime) for startTime in df['start time'][firstExp-2:firstExp]]
            dprime['control'].append(np.mean([obj.dprimeOtherModalGo for obj in controlSessions],axis=0))
            hitCount['control'].append(np.mean([obj.hitCount for obj in controlSessions],axis=0))


# opto stim plots
xticks = np.arange(len(stimNames))
for lbl in areaLabels:
    fig = plt.figure(figsize=(6,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        for opto,clr in zip(('no opto','opto'),'kb'):
            rr = respRate[lbl][goStim][opto]
            if i==0 and opto=='no opto':
                fig.suptitle(lbl + ' (n = ' + str(len(rr)) + ' mice)',fontsize=16)
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
for lbl in areaLabels:
    fig = plt.figure(figsize=(6,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        for opto,clr in zip(('no opto','opto'),'kb'):
            rr = respTime[lbl][goStim][opto]
            if i==0 and opto=='no opto':
                fig.suptitle(lbl+ ' (n = ' + str(len(rr)) + ' mice)',fontsize=16)
            if len(rr) > 0:
                mean = np.nanmean(rr,axis=0)[[0,2]]
                sem = np.nanstd(rr,axis=0)[[0,2]]/(len(rr)**0.5)
                ax.plot(xticks,mean,color=clr,lw=2,label=opto)
                for x,m,s in zip(xticks,mean,sem):
                    ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=14)
        ax.set_xticks(xticks)
        if i==1:
            ax.set_xticklabels(('vis1','sound1'))
        else:
            ax.set_xticklabels([])
        ax.set_xlim([-0.25,len(xticks)-0.75])
        ax.set_ylim([-0.1,0.3])
        if i == 0:
            ax.set_ylabel('Norm. response time (s)',fontsize=14)
        ax.set_title(goStim + ' rewarded',fontsize=14)
        if i==0:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=14)
    plt.tight_layout()
    

xticks = np.arange(len(stimNames))
for lbl in areaLabels:
    fig = plt.figure(figsize=(5,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        ax.plot([-1,6],[0,0],'k--')
        rr = [np.array(respRate[lbl][goStim][opto]) for opto in ('no opto','opto')]
        rr = rr[1] - rr[0]
        if i==0:
            fig.suptitle(lbl + ' (n = ' + str(len(rr)) + ' mice)')
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
for lbl in areaLabels:
    fig = plt.figure(figsize=(4,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        ax.plot([-1,6],[0,0],'k--')
        rr = [np.array(respRate[lbl][goStim][opto]) for opto in ('no opto','opto')]
        if len(rr[0]) > 0:
            rr = 100 * ((rr[1][:,[0,2]] - rr[0][:,[0,2]]) / rr[0][:,[0,2]])
            if i==0:
                fig.suptitle(lbl + ' (n = ' + str(len(rr)) + ' mice)')
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
    

# repeat vs non-repeat trials
for lbl in areaLabels:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = [0,1]
    ax.plot(alim,alim,'--',color='0.5')
    ax.plot(respRateRepeat[lbl]['no opto'],respRateRepeat[lbl]['opto'],'o',mec='k',mfc='k')
    ax.plot(respRateNonRepeat[lbl]['no opto'],respRateNonRepeat[lbl]['opto'],'o',mec='k',mfc='none')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('opto')
    ax.set_ylabel('no opto')
    ax.set_title(lbl)
    plt.tight_layout()
    
for lbl in areaLabels:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = [-4,4]
    ax.plot(alim,alim,'--',color='0.5')
    ax.plot(respTimeRepeat[lbl]['no opto'],respTimeRepeat[lbl]['opto'],'o',mec='k',mfc='k')
    ax.plot(respTimeNonRepeat[lbl]['no opto'],respTimeNonRepeat[lbl]['opto'],'o',mec='k',mfc='none')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('opto')
    ax.set_ylabel('no opto')
    ax.set_title(lbl)
    plt.tight_layout()


# opto feedback plots
x = np.arange(6) + 1
for lbl in dprime:
    n = len(dprime[lbl])
    if n > 0:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for optoBlock in (2,5):
            ax.add_patch(matplotlib.patches.Rectangle([optoBlock-0.4,0],width=0.8,height=5,facecolor=np.array([0,176,240])/255,edgecolor=None,alpha=0.5,zorder=0))
        m = np.mean(dprime[lbl],axis=0)
        s = np.std(dprime[lbl],axis=0) / (n**0.5)
        ax.plot(x,m,'k')
        ax.plot([x,x],[m-s,m+s],'k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim([0,3.5])
        ax.set_xlabel('Block')
        ax.set_ylabel('Cross modal d\'')
        ax.set_title(lbl+' (n='+str(n)+' mice)')
        plt.tight_layout()




















