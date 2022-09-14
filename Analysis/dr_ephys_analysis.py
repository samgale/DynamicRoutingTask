# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:51:54 2022

@author: svc_ccg
"""

import glob
import os
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import h5py
from sync import sync
import probeSync
import ecephys


def getSdf(spikes,startTimes,windowDur,sampInt=0.001,filt='exponential',filtWidth=0.005,avg=True):
        t = np.arange(0,windowDur+sampInt,sampInt)
        counts = np.zeros((startTimes.size,t.size-1))
        for i,start in enumerate(startTimes):
            counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,t)[0]
        if filt in ('exp','exponential'):
            filtPts = int(5*filtWidth/sampInt)
            expFilt = np.zeros(filtPts*2)
            expFilt[-filtPts:] = scipy.signal.exponential(filtPts,center=0,tau=filtWidth/sampInt,sym=False)
            expFilt /= expFilt.sum()
            sdf = scipy.ndimage.filters.convolve1d(counts,expFilt,axis=1)
        else:
            sdf = scipy.ndimage.filters.gaussian_filter1d(counts,filtWidth/sampInt,axis=1)
        if avg:
            sdf = sdf.mean(axis=0)
        sdf /= sampInt
        return sdf,t[:-1]


# behavior
behavPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\2022-08-17_13-25-06_626791\DynamicRouting1_626791_20220817_132623.hdf5"
d = h5py.File(behavPath,'r')

trialEndFrame = d['trialEndFrame'][:]
nTrials = trialEndFrame.size
trialStartFrame = d['trialStartFrame'][:nTrials]
stimStartFrame = d['trialStimStartFrame'][:nTrials]
trialStim = d['trialStim'].asstr()[:nTrials]
trialVisStimFrames = d['trialVisStimFrames'][:nTrials]
trialSoundDur = d['trialSoundDur'][:nTrials]
trialSoundArray = d['trialSoundArray'][:nTrials]
soundSampleRate = d['soundSampleRate'][()]
soundType = d['soundType'].asstr()[()]
if soundType == 'tone':
    param = 'toneFreq'
    toneFreq = {key: d[param][key][()] for key in d[param]}

d.close()


# rf mapping
rfMappingPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\2022-08-17_13-25-06_626791\RFMapping_test_20220817_142818.hdf5"
d = h5py.File(rfMappingPath,'r')

rfStimFrames = d['stimFrames'][()]
rfSoundDur = d['soundDur'][()]
rfSoundSampleRate = d['soundSampleRate'][()]
rfStimStartFrame = d['stimStartFrame'][()]
rfTrialVisXY = d['trialVisXY'][()]
rfTrialGratingOri = d['trialGratingOri'][()]
rfTrialSoundFreq = d['trialSoundFreq'][()]
rfTrialSoundArray = d['trialSoundArray'][()]

d.close()


# sync
syncPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\2022-08-17_13-25-06_626791\20220817T132454.h5"
syncDataset = sync.Dataset(syncPath)
    
vsyncRising,vsyncFalling = probeSync.get_sync_line_data(syncDataset,'vsync_stim')
vsyncTimes = vsyncFalling[1:] if vsyncFalling[0] < vsyncRising[0] else vsyncFalling

syncBarcodeRising,syncBarcodeFalling = probeSync.get_sync_line_data(syncDataset,'barcode_ephys')

syncBarcodeTimes,syncBarcodes = ecephys.extract_barcodes_from_times(syncBarcodeRising,syncBarcodeFalling)

firstRfFrame = np.where(np.diff(vsyncTimes)>10)[0][0] + 1


# ephys sync data
ephysPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\2022-08-17_13-25-06_626791\Record Node 108\experiment1\recording1"

ephysSampleRate = 30000

dirNames = ['ProbeA','ProbeB','ProbeC','ProbeF','DAQ']

syncData = {key: {} for key in dirNames}
for key in syncData:
    keyMod = '*'+key+'-AP' if 'Probe' in key else '*'+key+'*'  
    datTimestampsPath = os.path.join(glob.glob(os.path.join(ephysPath,'continuous',keyMod))[0],'sample_numbers.npy')
    ttlStatesPath = os.path.join(glob.glob(os.path.join(ephysPath,'events',keyMod))[0],'TTL','states.npy')
    ttlTimestampsPath = os.path.join(os.path.dirname(ttlStatesPath),'sample_numbers.npy')

    datTimestamps = np.load(datTimestampsPath) / ephysSampleRate
    
    ttlStates = np.load(ttlStatesPath)
    ttlTimestamps = np.load(ttlTimestampsPath) / ephysSampleRate - datTimestamps[0]
    
    barcodeRising = ttlTimestamps[ttlStates>0]
    barcodeFalling = ttlTimestamps[ttlStates<0]
    barcodeTimes,barcodes = ecephys.extract_barcodes_from_times(barcodeRising,barcodeFalling)
    
    shift,sampleRate,endpoints = ecephys.get_probe_time_offset(syncBarcodeTimes,syncBarcodes,barcodeTimes,barcodes,0,ephysSampleRate)

    syncData[key]['shift'] = shift
    syncData[key]['sampleRate'] = sampleRate


# analog data and stim latencies
nidaqDatPath = os.path.join(glob.glob(os.path.join(ephysPath,'continuous','*DAQ*'))[0],'continuous.dat')

numAnalogCh = 8
nidaqData = np.memmap(nidaqDatPath,dtype='int16',mode='r')    
nidaqData = np.reshape(nidaqData,(int(nidaqData.size/numAnalogCh),-1)).T

speakerCh = 1
microphoneCh = 3
microphoneData = nidaqData[microphoneCh]

stimLatency = np.zeros(nTrials)
preTime = 0.1
postTime = 0.1
for trial,stim in enumerate(trialStim):
    if 'sound' in stim:
        startFrame = stimStartFrame[trial]
        startTime = vsyncTimes[startFrame] + syncData['DAQ']['shift']
        soundDur = trialSoundDur[trial]
        startSample = int((startTime - preTime) * syncData['DAQ']['sampleRate'])
        endSample = int((startTime + soundDur + postTime) * syncData['DAQ']['sampleRate'])
        t = np.arange(endSample-startSample) / syncData['DAQ']['sampleRate'] - preTime
        sound = trialSoundArray[trial]
        tInterp = np.arange(-preTime,soundDur+postTime,1/soundSampleRate)
        mic = microphoneData[startSample:endSample]
        micInterp = np.interp(tInterp,t,mic)
        c = np.correlate(micInterp,sound,'valid')
        stimLatency[trial] = tInterp[np.argmax(c)]
    elif 'vis' in stim:
        stimLatency[trial] = 1.5/60 # approximately
        
rfStimLatency = np.zeros(rfStimStartFrame.size)
preTime = 0.1
postTime = 0.1
for trial,stim in enumerate(rfTrialSoundFreq):
    if np.isnan(stim):
        rfStimLatency[trial] = 1.5/60 # approximately
    else:
        startFrame = rfStimStartFrame[trial]
        startTime = vsyncTimes[firstRfFrame:][startFrame] + syncData['DAQ']['shift']
        startSample = int((startTime - preTime) * syncData['DAQ']['sampleRate'])
        endSample = int((startTime + rfSoundDur + postTime) * syncData['DAQ']['sampleRate'])
        t = np.arange(endSample-startSample) / syncData['DAQ']['sampleRate'] - preTime
        sound = rfTrialSoundArray[trial]
        tInterp = np.arange(-preTime,rfSoundDur+postTime,1/soundSampleRate)
        mic = microphoneData[startSample:endSample]
        micInterp = np.interp(tInterp,t,mic)
        c = np.correlate(micInterp,sound,'valid')
        rfStimLatency[trial] = tInterp[np.argmax(c)]


# unit data
unitData = {}
for probe in dirNames[:-1]:
    dirPath = glob.glob(os.path.join(ephysPath,'continuous','*'+probe+'-AP'))[0]
    kilosortData = {key: np.load(os.path.join(dirPath,key+'.npy')) for key in ('spike_clusters',
                                                                               'spike_times',
                                                                               'templates',
                                                                               'spike_templates',
                                                                               'channel_positions',
                                                                               'amplitudes')}
    clusterIDs = pd.read_csv(os.path.join(dirPath,'cluster_KSLabel.tsv'),sep='\t')
    unitIDs = np.unique(kilosortData['spike_clusters'])
    
    unitData[probe] = {}
    for u in unitIDs:
        uind = np.where(kilosortData['spike_clusters']==u)[0]
        u = str(u)
        unitData[probe][u] = {}
        unitData[probe][u]['label'] = clusterIDs[clusterIDs['cluster_id']==int(u)]['KSLabel'].tolist()[0]
        unitData[probe][u]['times'] = kilosortData['spike_times'][uind].flatten() / syncData[probe]['sampleRate'] - syncData[probe]['shift']
        
        #choose 1000 spikes with replacement, then average their templates together
        chosen_spikes = np.random.choice(uind,1000)
        chosen_templates = kilosortData['spike_templates'][chosen_spikes].flatten()
        unitData[probe][u]['template'] = np.mean(kilosortData['templates'][chosen_templates],axis=0)
        
        peakChan = np.unravel_index(np.argmin(unitData[probe][u]['template']),unitData[probe][u]['template'].shape)[1]
        unitData[probe][u]['peakChan'] = peakChan
        unitData[probe][u]['position'] = kilosortData['channel_positions'][peakChan]
        unitData[probe][u]['amplitudes'] = kilosortData['amplitudes'][uind]
        
        template = unitData[probe][u]['template'][:,peakChan]
        if any(np.isnan(template)):
            unitData[probe][u]['peakToTrough'] = np.nan
        else:
            peakInd = np.argmin(template)
            unitData[probe][u]['peakToTrough'] = np.argmax(template[peakInd:])/(syncData[probe]['sampleRate']/1000)


goodUnits = {}
unitYpos = {}
ephysDur = nidaqData.shape[1]/ephysSampleRate
minSpikeRate = 0.1
for probe in unitData:
    sortedUnits = np.array(list(unitData[probe].keys()))[np.argsort([unitData[probe][u]['peakChan'] for u in unitData[probe]])]
    goodUnits[probe] = [u for u in sortedUnits if unitData[probe][u]['label']=='good' and len(unitData[probe][u]['times'])/ephysDur > minSpikeRate]
    unitYpos[probe] = [unitData[probe][u]['position'][1] for u in goodUnits[probe]]
    

# rf mapping sdfs and spike count
xy = [tuple(p) for p in np.unique(rfTrialVisXY[~np.isnan(rfTrialVisXY[:,0])],axis=0)]
freq = list(np.unique(rfTrialSoundFreq[~np.isnan(rfTrialSoundFreq)]))
paramNames = ('sdfs','spikeCount','firstSpikeLat','hasResp')
rfVisData = {probe: {stim: {param: [] for param in paramNames} for stim in xy} for probe in unitData}
rfSoundData = {probe: {stim: {param: [] for param in paramNames} for stim in freq} for probe in unitData}
preTime = 0.1
postTime = 0.4
for probe in unitData:
    for stim in xy + freq:
        if isinstance(stim,tuple):
            d = rfVisData
            trials = np.all(rfTrialVisXY==stim,axis=1)
        else:
            d = rfSoundData
            trials = rfTrialSoundFreq==stim
        startTimes = vsyncTimes[firstRfFrame:][rfStimStartFrame[trials]] + rfStimLatency[trials]
        for u in goodUnits[probe]:
            spikeTimes = unitData[probe][u]['times']
            s,sdfTime = getSdf(spikeTimes,startTimes-preTime,preTime+postTime)
            d[probe][stim]['sdfs'].append(s)
            preSpikes = []
            postSpikes = []
            lat = []
            for t in startTimes:
                st = spikeTimes[(spikeTimes>t-0.1) & (spikeTimes<t+0.1)]
                preSpikes.append(np.sum(st<t))
                postSpikes.append(np.sum(st>t))
                firstSpike = np.where((st>t+0.01) & (st<t+0.1))[0]
                if len(firstSpike)>0:
                    lat.append(st[firstSpike[0]]-t)
                else:
                    lat.append(np.nan)
            d[probe][stim]['spikeCount'].append(np.mean(postSpikes)-np.mean(preSpikes))
            d[probe][stim]['firstSpikeLat'].append(np.nanmedian(lat)*1000)
            pval = 1 if np.sum(np.array(postSpikes)-np.array(preSpikes))==0 else scipy.stats.wilcoxon(preSpikes,postSpikes)[1] 
            bs = s - s[sdfTime<preTime].mean()
            z = bs / s[sdfTime<preTime].std()
            d[probe][stim]['hasResp'].append(pval < 0.05 and z[(sdfTime>preTime) & (sdfTime<preTime+0.1)].max() > 5)


hasVisResp = {}
hasSoundResp = {}
for d,r in zip((rfVisData,rfSoundData),(hasVisResp,hasSoundResp)):
    for probe in d:
        r[probe] = np.any(np.stack([d[probe][stim]['hasResp'] for stim in d[probe]],axis=1),axis=1)
        
        
# resp along length of probe
fig = plt.figure(figsize=(5,8))
for i,probe in enumerate(unitData):
    ax = fig.add_subplot(len(unitData),1,i+1)
    for d,clr,lbl in zip((rfVisData,rfSoundData),'gm',('vis','aud')):
        r = np.max(np.stack([d[probe][stim]['spikeCount'] for stim in d[probe]],axis=1),axis=1)
        ax.plot(unitYpos[probe],r,'o',mec=clr,mfc='none',alpha=0.5,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    if i==len(unitData)-1:
        ax.set_xlabel('unit y position')
    if i==0:
        ax.set_ylabel('response (spikes)')
        ax.legend()
    ax.set_title(probe)
plt.tight_layout()


# latency
fig = plt.figure(figsize=(5,8))
for i,probe in enumerate(unitData):
    ax = fig.add_subplot(len(unitData),1,i+1)
    for d,hasResp,clr,lbl in zip((rfVisData,rfSoundData),(hasVisResp,hasSoundResp),'gm',('vis','aud')):
        maxRespInd = np.argmax(np.stack([d[probe][stim]['spikeCount'] for stim in d[probe]],axis=1),axis=1)
        lat = np.stack([d[probe][stim]['firstSpikeLat'] for stim in d[probe]],axis=1)[np.arange(maxRespInd.size),maxRespInd]
        lat = lat[~np.isnan(lat) & hasResp[probe]]
        latSort = np.sort(lat)
        cumProb = np.array([np.sum(lat<=s)/lat.size for s in latSort])
        ax.plot(latSort,cumProb,color=clr,label=lbl+', n='+str(lat.size))
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_xlim([10,100])
    if i==len(unitData)-1:
        ax.set_xlabel('first spike latency (ms)')
    if i==0:
        ax.set_ylabel('cumulative fraction of units')
    ax.legend()
    ax.set_title(probe)
plt.tight_layout()


# response to rf mapping visual stimuli
xpos = np.unique(np.array(xy)[:,0])
ypos = np.unique(np.array(xy)[:,1])
rfMap = {}
for probe in unitData:
    fig = plt.figure(figsize=(6,10))
    fig.suptitle(probe+' ('+str(hasVisResp[probe].sum())+' visual responsive units of '+str(len(goodUnits[probe]))+' good units)')
    gs = matplotlib.gridspec.GridSpec(ypos.size*2,xpos.size)
    axs = []
    ymax = 0
    rfMap[probe] = np.zeros((len(goodUnits[probe]),ypos.size,xpos.size))
    for pos in xy:
        x,y = pos
        i = ypos.size-1-np.where(ypos==y)[0][0]
        j = np.where(xpos==x)[0][0]
        rfMap[probe][:,i,j] = rfVisData[probe][pos]['spikeCount']
        ax = fig.add_subplot(gs[i,j])
        meanSdf = np.mean(np.array(rfVisData[probe][pos]['sdfs'])[hasVisResp[probe]],axis=0)
        ax.plot(sdfTime-preTime,meanSdf,'k')
        ymax = max(ymax,meanSdf.max())
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        if i<ypos.size-1:
            ax.set_xticklabels([])
        elif j==0:
            ax.set_xlabel('time from stim onset (s)')
        if j>0:
            ax.set_yticklabels([])
        elif i==0:
            ax.set_ylabel('spikes/s')
        ax.set_title(np.round(pos).astype(int),fontsize=8)
        axs.append(ax)
    for ax in axs:
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0,1.02*ymax])
    
    ax = fig.add_subplot(gs[ypos.size:,:xpos.size])
    ax.imshow(rfMap[probe][hasVisResp[probe]].mean(axis=0),cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    

for probe in unitData:
    fig = plt.figure(figsize=(8,10))
    fig.suptitle(probe)
    ypos = np.array(unitYpos[probe])
    h = np.arange(np.floor(ypos.min()/100)*100,np.ceil(ypos.max()/100)*100+1,100)
    yi = np.searchsorted(h,ypos)
    w = [np.sum(yi==k) for k in np.unique(yi)]
    j = np.zeros(h.size).astype(int)
    gs = matplotlib.gridspec.GridSpec(h.size,max(w))
    for rf,i in zip(rfMap[probe],yi):
        ax = fig.add_subplot(gs[i,j[i]])
        j[i] += 1
        ax.imshow(rf,cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    
    
# ori
ori = np.unique(rfTrialGratingOri[~np.isnan(rfTrialGratingOri)])
rfOriResp = {probe: {stim: [] for stim in ori} for probe in unitData}
for probe in unitData:
    for stim in ori:
        trials = rfTrialGratingOri==stim
        startTimes = vsyncTimes[firstRfFrame:][rfStimStartFrame[trials]] + rfStimLatency[trials]
        for u in goodUnits[probe]:
            spikeTimes = unitData[probe][u]['times']
            preSpikes = []
            postSpikes = []
            for t in startTimes:
                preSpikes.append(np.sum((spikeTimes>t-0.1) & (spikeTimes<t)))
                postSpikes.append(np.sum((spikeTimes>t) & (spikeTimes<t+0.1)))
            rfOriResp[probe][stim].append(np.mean(postSpikes)-np.mean(preSpikes))
            
fig = plt.figure(figsize=(10,10))
for i,probe in enumerate(rfOriResp):
    r = np.stack([rfOriResp[probe][stim] for stim in rfOriResp[probe]],axis=0)
    ax = fig.add_subplot(len(unitData),1,i+1)
    im = ax.imshow(r[::-1],cmap='gray')
    ax.set_yticks([0,len(ori)-1])
    ax.set_yticklabels(ori[[-1,0]])
    if i==len(unitData)-1:
        ax.set_xlabel('unit')
    if i==0:
        ax.set_ylabel('ori (deg)')
    ax.set_title(probe)
    cb = plt.colorbar(im,ax=ax,fraction=0.005,pad=0.01)
    
        
# response to rf mapping auditory stimuli
for probe in unitData:
    fig = plt.figure(figsize=(6,10))
    fig.suptitle(probe+' ('+str(hasSoundResp[probe].sum())+' auditory responsive units of '+str(len(goodUnits[probe]))+' good units)')
    axs = []
    ymax = 0
    for f in freq:
        i = freq[::-1].index(f)
        ax = fig.add_subplot(len(freq),1,i+1)
        meanSdf = np.mean(np.array(rfSoundData[probe][f]['sdfs'])[hasSoundResp[probe]],axis=0)
        ax.plot(sdfTime-preTime,meanSdf,'k')
        ymax = max(ymax,meanSdf.max())
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        if i==len(freq)-1:
            ax.set_xlabel('time from stim onset (s)')
        else:
            ax.set_xticklabels([])
        if i==0:
            ax.set_ylabel('spikes/s')
        ax.set_title(str(int(f))+' Hz',fontsize=8)
        axs.append(ax)
    for ax in axs:
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0,1.02*ymax])
    plt.tight_layout()
    


fig = plt.figure(figsize=(10,10))
for i,probe in enumerate(unitData):
    r = np.stack([rfSoundData[probe][stim]['spikeCount'] for stim in rfSoundData[probe]],axis=0)
    ax = fig.add_subplot(len(unitData),1,i+1)
    im = ax.imshow(r[::-1],cmap='gray')
    ax.set_yticks([0,len(freq)-1])
    ax.set_yticklabels(np.array(freq)[[-1,0]].astype(int)/1000)
    if i==len(unitData)-1:
        ax.set_xlabel('unit')
    if i==0:
        ax.set_ylabel('frequency (kHz)')
    ax.set_title(probe)
    cb = plt.colorbar(im,ax=ax,fraction=0.005,pad=0.01)





