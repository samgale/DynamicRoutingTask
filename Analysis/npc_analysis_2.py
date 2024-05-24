#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import npc_lims
import npc_sessions

# %matplotlib widget

#%%
filePath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_707969_20240520\DynamicRouting1_707969_20240520_165134.hdf5"
#filePath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_703880_20240412\DynamicRouting1_703880_20240412_100321.hdf5"
session = npc_sessions.Session(filePath)

obj = session._trials

#%%
sync = session.sync_data
lineLabels = sync.line_labels

vsyncs = sync.get_falling_edges('vsync_stim','seconds')

stimRunning =  sync.get_rising_edges('stim_running','seconds')

#%%
trialStartTime = obj.quiescent_stop_time

stimLatency = obj.stim_start_time - trialStartTime

optoLatency = obj.opto_start_time - trialStartTime

optoStimOffset = optoLatency - stimLatency

#%%
fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(3,2)
bins = np.arange(-70,70)/1000
for i,(t,xlbl) in enumerate(zip((stimLatency,optoLatency,optoStimOffset),
                               ('stimulus latency','opto latency','opto offset'))):
    for j,(stimTrials,stim) in enumerate(zip((obj.is_vis_stim,obj.is_aud_stim),
                                             ('visual','auditory'))):
        ax = fig.add_subplot(gs[i,j])
        for trials,clr,lbl in zip((stimTrials & ~obj.is_opto,stimTrials & obj.is_opto),'kb',('no opto','opto')):
            if i > 0 and lbl == 'no opto':
                continue
            h,b = np.histogram(t[trials],bins=bins)
            ax.plot(bins[:-1],h,color=clr,label=lbl)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlabel(xlbl)
        if i==1 and j==0:
            ax.set_ylabel('# trials')
        if i==0:
            ax.set_title(stim)
        if i==0 and j==1:
            ax.legend(loc='upper right')
plt.tight_layout()

#%%
fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(2,2)
bins = np.arange(55)/1000
trialStartIndex = np.array([np.where(obj._flip_times==t)[0][0] for t in trialStartTime])
pre = trialStartTime - obj._flip_times[trialStartIndex-1]
post = obj._flip_times[trialStartIndex+1] - trialStartTime
for i,(stimTrials,stim) in enumerate(zip((obj.is_vis_stim,obj.is_aud_stim),
                                         ('visual','auditory'))):
    for j,t in enumerate((pre,post)):
        ax = fig.add_subplot(gs[i,j])
        for trials,clr,lbl in zip((stimTrials & ~obj.is_opto,stimTrials & obj.is_opto),'kb',('no opto','opto')):
            h,b = np.histogram(t[trials],bins=bins)
            ax.plot(bins[:-1],h,color=clr,label=lbl)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        if i==1:
            if j==0:
                ax.set_xlabel('Frame interval before trial start (s)')
            else:
                ax.set_xlabel('Frame interval after trial start (s)')
        if j==0:
            ax.set_ylabel('# trials')
        if i==0 and j==1:
            ax.legend(loc='upper right')
        ax.set_title(stim)
plt.tight_layout()






















# %%
