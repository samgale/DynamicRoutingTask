#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import npc_lims
import npc_sessions

#%%
session = npc_sessions.Session('689663 20240403')

obj = session._trials

#%%
trialStartTime = obj.quiescent_stop_time

stimLatency = obj.stim_start_time - trialStartTime

optoLatency = obj.opto_start_time - trialStartTime

optoStimOffset = stimLatency - optoLatency

#%%
fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(3,2)
bins = np.arange(-60,60)/1000
for i,(t,lbl) in enumerate(zip((stimLatency,optoLatency,optoStimOffset),
                               ('stimulus latency','opto latency','opto offset'))):
    for j,(stimTrials,stim) in enumerate(zip((obj.is_vis_stim,obj.is_aud_stim),
                                             ('visual','auditory'))):
        ax = fig.add_subplot(gs[i,j])
        trials = stimTrials & obj.is_opto if i>0 else stimTrials
        ax.hist(t[trials],bins=bins,color='k')
        ax.set_xlabel(lbl)
        if i==0:
            ax.set_title(stim)
        if i==1 and j==0:
            ax.set_ylabel('# of trials')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
plt.tight_layout()

#%%
fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(2,2)
bins = np.arange(100)/1000
trialStartIndex = np.array([np.where(obj._flip_times==t)[0][0] for t in trialStartTime])
pre = trialStartTime - obj._flip_times[trialStartIndex-2]
post = obj._flip_times[trialStartIndex+2] - trialStartTime
for i,(stimTrials,stim) in enumerate(zip((obj.is_vis_stim,obj.is_aud_stim),
                                         ('visual','auditory'))):
    for j,(t,lbl) in enumerate(zip((pre,post),('pre','post'))):
        ax = fig.add_subplot(gs[i,j])
        trials = stimTrials & obj.is_opto if i>0 else stimTrials
        for trials,clr in zip((stimTrials & ~obj.is_opto,stimTrials & obj.is_opto),'kb'):
            h,b = np.histogram(t[trials],bins=bins)
            ax.plot(bins[:-1],h,color=clr)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
plt.tight_layout()






















# %%
