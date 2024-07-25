import copy
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from DynamicRoutingAnalysisUtils import getSessionData


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

optoExps = pd.read_excel(os.path.join(baseDir,'Sam','OptoExperiments.xlsx'),sheet_name=None)


areaNames = ('V1','PPC','pACC','aACC','plFC','mFC','lFC')
areaLabels = (('V1','V1 left'),('PPC',),('pACC',),('aACC','ACC'),('plFC',),('mFC',),('lFC','PFC'))
stimNames = ('vis1','vis2','sound1','sound2','catch')
respRate = {area: {goStim: {opto: [] for opto in ('no opto','opto')} for goStim in ('vis1','sound1')} for area in areaNames}
for mid in optoExps.keys():
    df = optoExps[mid] 
    exps = [getSessionData(mid,startTime) for startTime in df['session start time']]
    labels = np.unique(np.concatenate([obj.optoParams['label'] for obj in exps]))
    
    for area,lbl in zip(areaNames,areaLabels):
        if np.any(df[area]) and np.any(np.in1d(labels,lbl)):
            for i,goStim in enumerate(('vis1','sound1')):
                for optoLbl in ('no opto',lbl):
                    r = np.zeros(len(stimNames))
                    n = r.copy()
                    for obj in [exp for exp,hasArea in zip(exps,df[area]) if hasArea]:
                        blockTrials = (obj.rewardedStim==goStim) & ~obj.autoRewardScheduled
                        optoTrials = obj.trialOptoLabel=='no opto' if optoLbl=='no opto' else np.in1d(obj.trialOptoLabel,lbl)
                        for j,stim in enumerate(stimNames):
                            trials = blockTrials & optoTrials & (obj.trialStim==stim)
                            r[j] += obj.trialResponse[trials].sum()
                            n[j] += trials.sum()
                    respRate[area][goStim]['no opto' if optoLbl=='no opto' else 'opto'].append(r/n)


xticks = np.arange(len(stimNames))
for area in areaNames:
    fig = plt.figure(figsize=(6,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        for opto,clr in zip(('no opto','opto'),'kb'):
            rr = respRate[area][goStim][opto]
            if i==0 and opto=='no opto':
                fig.suptitle(area + ' (n = ' + str(len(rr)) + ' mice)')
            mean = np.mean(rr,axis=0)
            sem = np.std(rr,axis=0)/(len(rr)**0.5)
            ax.plot(xticks,mean,color=clr,lw=2,label=opto)
            for x,m,s in zip(xticks,mean,sem):
                ax.plot([x,x],[m-s,m+s],color=clr,lw=2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(xticks)
        if i==1:
            ax.set_xticklabels(stimNames)
        else:
            ax.set_xticklabels([])
        ax.set_xlim([-0.25,len(stimNames)-0.75])
        ax.set_ylim([-0.01,1.01])
        ax.set_ylabel('Response Rate')
        ax.set_title(goStim + ' rewarded')
        if i==0:
            ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()
    

xticks = np.arange(len(stimNames))
for area in areaNames:
    fig = plt.figure(figsize=(4,5))
    for i,goStim in enumerate(('vis1','sound1')):
        ax = fig.add_subplot(2,1,i+1)
        ax.plot([-1,6],[0,0],'k--')
        rr = [np.array(respRate[area][goStim][opto]) for opto in ('no opto','opto')]
        rr = rr[1] - rr[0]
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
        rr = [np.array(respRate[area][goStim][opto])[:,[0,2]] for opto in ('no opto','opto')]
        rr = 100 * ((rr[1] - rr[0]) / rr[0])
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























