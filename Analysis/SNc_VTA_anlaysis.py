#%%
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import npc_lims
import npc_sessions

#%%
def getGoodUnits(df):
    return df.query("isi_violations_ratio <= 0.5 & activity_drift <= 0.2 & presence_ratio >= 0.7 & amplitude_cutoff <= 0.1 & decoder_label != 'noise'")

def getAlignedSpikes(spikeTimes,startTimes,windowDur,binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    spikes = np.zeros((len(startTimes),bins.size-1))
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikeTimes,start)
        endInd = np.searchsorted(spikeTimes,start+windowDur)
        spikes[i] = np.histogram(spikeTimes[startInd:endInd]-start, bins)[0]
    return spikes

#%%
allUnitsDf = pd.read_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.268/consolidated/units.parquet')

areas = ('SNc','VTA')

sessionIds = allUnitsDf.query('structure.isin(@areas)')['session_id'].unique()

nUnits = [getGoodUnits(allUnitsDf.query('session_id==@sid & structure.isin(@areas)')).shape[0] for sid in sessionIds]

nSessions = np.sum(np.array(nUnits)>0)

#%%
unitsDf = pd.read_parquet(r"\\allen\programs\mindscope\workgroups\dynamicrouting\ben\SNc_VTA_waveforms.parquet")
unitsDf = getGoodUnits(unitsDf)

#%%
trialsDf = pd.read_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.268/consolidated/trials.parquet')

#%%
minTrials = 1
binSize = 0.05
preTime = 0.5
windowDur = preTime + 1
t = np.arange(0,windowDur+binSize,binSize)[:-1] + binSize/2
blockType = ('vis rewarded','aud rewarded')
stimType = ('vis target','aud target')
respType = ('resp','no resp')
psth = {block: {stim: {resp: [] for resp in respType} for stim in stimType} for block in blockType}
for u,s in zip(unitsDf['unit_id'],unitsDf['session_id']):
    spikeTimes = unitsDf.query('unit_id==@u')['spike_times'].iloc[0]
    trials = trialsDf.query("session_id==@s")
    isResp = trials['is_response']
    isVisTarg = trials['is_vis_target']
    isAudTarg = trials['is_aud_target']
    isVisRew = trials['is_vis_rewarded']
    isAudRew = trials['is_aud_rewarded']
    n = []
    for block in blockType:
        for stim in stimType:
            if ('vis' in block and 'aud' in stim) or ('aud' in block and 'vis' in stim):
                for resp in respType:
                    isTrial = (isVisRew if 'vis' in block else isAudRew) & (isVisTarg if 'vis' in stim else isAudTarg) & (isResp if resp=='resp' else ~isResp)
                    n.append(isTrial.sum())
    if np.any(np.array(n) < minTrials):
        continue
    for block in blockType:
        for stim in stimType:
            for resp in respType:
                isTrial = (isVisRew if 'vis' in block else isAudRew) & (isVisTarg if 'vis' in stim else isAudTarg) & (isResp if resp=='resp' else ~isResp)
                if isTrial.sum() < minTrials:
                    psth[block][stim][resp].append(np.full(t.size,np.nan))
                else: 
                    startTimes =  trials[isTrial]['stim_start_time']
                    psth[block][stim][resp].append(np.mean(getAlignedSpikes(spikeTimes,startTimes-preTime,windowDur,binSize),axis=0))

#%%
for stim,clr in zip(stimType,'rb'):
    plt.figure()
    for block in blockType:
        alpha = 1 if ('vis' in block and 'vis' in stim) or ('aud' in block and 'aud' in stim) else 0.5
        for resp in respType:
            ls = '-' if resp=='resp' else ':'
            plt.plot(t,np.nanmean(psth[block][stim][resp],axis=0)/binSize,color=clr,ls=ls,alpha=alpha)





# %%
