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

#%%
areas = ('SNc','VTA')

areas = ('SNr',)

unitsDf = allUnitsDf.query('structure.isin(@areas)')

sessionIds = unitsDf['session_id'].unique()

nUnits = [getGoodUnits(allUnitsDf.query('session_id==@sid & structure.isin(@areas)')).shape[0] for sid in sessionIds]

nSessions = np.sum(np.array(nUnits)>0)

#%%
unitsDf = pd.read_parquet(r"\\allen\programs\mindscope\workgroups\dynamicrouting\ben\SNc_VTA_waveforms.parquet")
unitsDf = getGoodUnits(unitsDf)

#%%
trialsDf = pd.read_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.268/consolidated/trials.parquet')

#%%
fig = plt.figure()
ax = fig.add_subplot()
wf = np.stack(unitsDf['waveform_mean'])
t = np.arange(wf.shape[1])/30000
# for w in wf:
ax.plot(t,wf[2],'k',alpha=0.25)
ax.set_xlim([0.002,0.006])



#%%
minTrials = 1
nTrialsAvg = 1
binSize = 0.1
preTime = 0.5
windowDur = preTime + 1
t = np.arange(0,windowDur+binSize,binSize)[:-1] + binSize/2
blockType = ('vis rewarded','aud rewarded')
stimType = ('vis target','aud target')
trialType = ('previous','first','last')
respType = ('resp','no resp')
psth = {block: {stim: {trial: {resp: [] for resp in respType} for trial in trialType} for stim in stimType} for block in blockType}
for u,s in zip(unitsDf['unit_id'],unitsDf['session_id']):
    spikeTimes = unitsDf.query('unit_id==@u')['spike_times'].iloc[0]
    trials = trialsDf.query("session_id==@s")
    isResp = trials['is_response']
    isVisTarg = trials['is_vis_target']
    isAudTarg = trials['is_aud_target']
    isVisRew = trials['is_vis_rewarded']
    isAudRew = trials['is_aud_rewarded']
    for block in blockType:
        blockTrials = isVisRew if 'vis' in block else isAudRew
        for stim in stimType:
            stimTrials = isVisTarg if 'vis' in stim else isAudTarg
            for trial in trialType:
                isTrial = np.zeros(stimTrials.size,dtype=bool)
                blks = range(5) if trial=='previous' else range(1,6)
                i = slice(0,nTrialsAvg) if trial=='first' else slice(-nTrialsAvg,None)
                isTrial[[np.where(stimTrials & (trials['block_index']==b))[0][i] for b in blks]] = True
                for resp in respType:
                    ind = blockTrials & isTrial & (isResp if resp=='resp' else ~isResp)
                    if ind.sum() < minTrials:
                        psth[block][stim][trial][resp].append(np.full(t.size,np.nan))
                    else: 
                        startTimes =  trials[ind]['response_time']
                        psth[block][stim][trial][resp].append(np.mean(getAlignedSpikes(spikeTimes,startTimes-preTime,windowDur,binSize),axis=0))

#%%
for trial in trialType:
    for stim,clr in zip(stimType,'rb'):
        plt.figure()
        for block in blockType:
            alpha = 1 if ('vis' in block and 'vis' in stim) or ('aud' in block and 'aud' in stim) else 0.5
            for resp in respType:
                ls = '-' if resp=='resp' else ':'
                plt.plot(t,np.nanmean(psth[block][stim][trial][resp],axis=0)/binSize,color=clr,ls=ls,alpha=alpha)





# %%
