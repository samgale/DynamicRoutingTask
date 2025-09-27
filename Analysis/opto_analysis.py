import copy
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getSessionData,getFirstExperimentSession,calcDprime


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

optoExps = pd.read_excel(os.path.join(baseDir,'Sam','OptoExperiments.xlsx'),sheet_name=None)

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

optoCoords = {'V1': (-3.5,2.6),
              'PPC': (-2.0,1.7),
              'RSC': (-2.5,0.5),
              'pACC': (-0.5,0.5),
              'aACC': (1.0,0.5),
              'plFC': (1.0,2.0),
              'mFC': (2.5,0.5),
              'lFC': (2.5,2.0)}

genotype = 'VGAT-ChR2' # VGAT-ChR2 or wt control
epoch = 'stim' # stim or feedback
hemi = 'bilateral' # unilateral, bilateral, or multilateral
hitThresh = 10

mice = []
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
    nTrials = copy.deepcopy(respRate)
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
        sessions = sessions & df['bilateral'] & ~df['unilateral']
    elif hemi == 'multilateral':
        sessions = sessions & df['unilateral'] & df['bilateral']
    sessions = sessions & np.any(np.stack([df[area] for area in areaNames],axis=1),axis=1)
    if np.any(sessions):
        mice.append(mid)
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
                            nTrials[lbl][goStim][optoKey].append(n)
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


# change in response rate map
respRateDiff = np.array([[np.mean((np.array(respRate[lbl][goStim]['opto']) - np.array(respRate[lbl][goStim]['no opto'])) / np.array(respRate[lbl][goStim]['no opto']),axis=0) for goStim in respRate[lbl]] for lbl in respRate])

rrDiffNorm = (respRateDiff + 1) / 2

sm = matplotlib.cm.ScalarMappable(cmap='bwr')
sm.set_array(rrDiffNorm)
sm.set_clim((-100,100))

for j,blockType in enumerate(('Vis rewarded','Aud rewarded')):
    for k,stim in zip((0,2),('Vis target','Aud target')):
        fig = plt.figure()
        ax = fig.add_subplot()
        for i,lbl in enumerate(areaLabels):
            y,x = optoCoords[lbl]
            for hemi in (-1,):
                ax.add_patch(matplotlib.patches.Circle((x*hemi,y),radius=0.5,color=matplotlib.cm.bwr(rrDiffNorm[i,j,k])))
        ax.set_xlim([-3.5,0.5])
        ax.set_ylim([-4.5,3.5])
        ax.set_aspect('equal')
        cb = plt.colorbar(sm,ax=ax,fraction=0.026,pad=0.04)
        ax.set_xlabel('Lateral from midline (mm)')
        ax.set_ylabel('Anterior from bregma (mm)')
        ax.set_title('% change in response rate (opto - no opto)'+'\n'+blockType+' block'+'\n'+stim)
    

# dprime
dprime = {lbl: {goStim: {opto: [] for opto in respRate[lbl][goStim]} for goStim in respRate[lbl]} for lbl in respRate}
for lbl in dprime:
    for goStim in dprime[lbl]:
        i = [0,2] if goStim=='vis1' else [2,0]
        for opto in dprime[lbl][goStim]:
            for r,n in zip(respRate[lbl][goStim][opto],nTrials[lbl][goStim][opto]):
                hr,fr = r[i]
                hn,fn = n[i]
                # dprime[lbl][goStim][opto].append(calcDprime(hr,fr,hn,fn))
                dprime[lbl][goStim][opto].append(calcDprime(hr,fr,50,50))
            
            # pooled
            # n = np.stack(nTrials[lbl][goStim][opto]) 
            # r = np.stack(respRate[lbl][goStim][opto])
            # r = np.sum(r*n,axis=0) / np.sum(n,axis=0)
            # n = np.sum(n,axis=0)
            # hr,fr = r[i]
            # hn,fn = n[i]
            # # dprime[lbl][goStim][opto].append(calcDprime(hr,fr,hn,fn))
            # dprime[lbl][goStim][opto].append(calcDprime(hr,fr,50,50))

for lbl in areaLabels:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    alim = [-5,5]
    ax.plot(alim,alim,'--',color='0.5')
    for goStim,clr in zip(dprime[lbl],'gm'):
        ax.plot(dprime[lbl][goStim]['no opto'],dprime[lbl][goStim]['opto'],'o',mec=clr,mfc=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('no opto')
    ax.set_ylabel('opto')
    ax.set_title(lbl)
    plt.tight_layout()
    
dpDiff = np.array([[np.median(np.array(dprime[lbl][goStim]['opto']) - np.array(dprime[lbl][goStim]['no opto'])) for goStim in dprime[lbl]] for lbl in dprime])
cmax = np.absolute(dpDiff).max()
dpDiffNorm = dpDiff / cmax
dpDiffNorm += 1
dpDiffNorm /= 2

sm = matplotlib.cm.ScalarMappable(cmap='bwr')
sm.set_array(dpDiffNorm)
sm.set_clim((-cmax,cmax))

for j,goStim in enumerate(('vis1','sound1')):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i,lbl in enumerate(areaLabels):
        y,x = optoCoords[lbl]
        for hemi in (-1,1):
            ax.add_patch(matplotlib.patches.Circle((x*hemi,y),radius=0.5,color=matplotlib.cm.bwr(dpDiffNorm[i,j])))
    ax.set_xlim([-3.5,3.5])
    ax.set_ylim([-4.5,3.5])
    ax.set_aspect('equal')
    cb = plt.colorbar(sm,ax=ax,fraction=0.026,pad=0.04)


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
    ax.set_xlabel('no opto')
    ax.set_ylabel('opto')
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
    ax.set_xlabel('no opto')
    ax.set_ylabel('opto')
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




















