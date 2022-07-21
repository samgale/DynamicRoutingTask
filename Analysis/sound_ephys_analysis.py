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


# behavior/stimuli
behavPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625821_07112022\DynamicRouting1_test_20220711_144242.hdf5"
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
soundType = {key: d['soundType'][key].asstr()[()] for key in d['soundType']}
soundParam = {key: d[param][key][()] for param in ('toneFreq','linearSweepFreq','logSweepFreq','noiseFiltFreq','ampModFreq')
                                     for key in d[param]}

d.close()


# sync
syncPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625821_07112022\20220711T144211.h5"
syncDataset = sync.Dataset(syncPath)
    
vsyncRising,vsyncFalling = probeSync.get_sync_line_data(syncDataset,'vsync_stim')
vsyncTimes = vsyncFalling[1:] if vsyncFalling[0] < vsyncRising[0] else vsyncFalling

syncBarcodeRising,syncBarcodeFalling = probeSync.get_sync_line_data(syncDataset,'barcode_ephys')

syncBarcodeTimes,syncBarcodes = ecephys.extract_barcodes_from_times(syncBarcodeRising,syncBarcodeFalling)


# ephys sync data
ephysPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625821_07112022\2022-07-11_14-42-15\Record Node 103\experiment1\recording1"

probeNames = ['B','C']
probeDirNames = ['Neuropix-PXI-100.2','Neuropix-PXI-100.4']
nidaqDirName = 'NI-DAQmx-101.0'

syncData = {key: {'dirName': dirName} for key,dirName in zip(probeNames+['nidaq'],probeDirNames+[nidaqDirName])}

ephysSampleRate = 30000
for key,d in syncData.items():
    datTimestampsPath = os.path.join(ephysPath,'continuous',d['dirName'],'timestamps.npy')
    ttlStatesPath = os.path.join(glob.glob(os.path.join(ephysPath,'events',d['dirName'],'TTL_*'))[0],'channel_states.npy')
    ttlTimestampsPath = os.path.join(os.path.dirname(ttlStatesPath),'timestamps.npy')

    datTimestamps = np.load(datTimestampsPath)
    
    ttlStates = np.load(ttlStatesPath)
    ttlTimestamps = np.load(ttlTimestampsPath) - datTimestamps[0]
    
    barcodeRising = ttlTimestamps[ttlStates>0] / ephysSampleRate
    barcodeFalling = ttlTimestamps[ttlStates<0] / ephysSampleRate
    barcodeTimes,barcodes = ecephys.extract_barcodes_from_times(barcodeRising,barcodeFalling)
    
    shift,sampleRate,endpoints = ecephys.get_probe_time_offset(syncBarcodeTimes,syncBarcodes,barcodeTimes,barcodes,0,ephysSampleRate)

    syncData[key]['shift'] = shift
    syncData[key]['sampleRate'] = sampleRate


# analog data and stim latencies
nidaqDatPath = os.path.join(ephysPath,'continuous',nidaqDirName,'continuous.dat')

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
        startTime = vsyncTimes[startFrame] + syncData['nidaq']['shift']
        stimDur = trialSoundDur[trial]
        startSample = int((startTime - preTime) * syncData['nidaq']['sampleRate'])
        endSample = int((startTime + stimDur + postTime) * syncData['nidaq']['sampleRate'])
        t = np.arange(endSample-startSample) / syncData['nidaq']['sampleRate'] - preTime
        sound = trialSoundArray[trial]
        tInterp = np.arange(-preTime,stimDur+postTime,1/soundSampleRate)
        mic = microphoneData[startSample:endSample]
        micInterp = np.interp(tInterp,t,mic)
        c = np.correlate(micInterp,sound,'valid')
        stimLatency[trial] = tInterp[np.argmax(c)]
    elif 'vis' in stim:
        stimLatency[trial] = 1.5/60 # approximately


# unit data
unitData = {}
for probe,dirName in zip(probeNames,probeDirNames):
    dirPath = os.path.join(ephysPath,'continuous',dirName)
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
ephysDur = nidaqData.shape[1]/ephysSampleRate
minSpikeRate = 0.1
for probe in probeNames:
    sortedUnits = np.array(list(unitData[probe].keys()))[np.argsort([unitData[probe][u]['peakChan'] for u in unitData[probe]])]
    goodUnits[probe] = [u for u in sortedUnits if unitData[probe][u]['label']=='good' and len(unitData[probe][u]['times'])/ephysDur > minSpikeRate]


# population response
stimNames = ['catch','vis1','vis2']+['sound'+str(i+1) for i in range(10)]
sdfs = {probe: {stim: [] for stim in stimNames} for probe in probeNames}
preStimTime = 0.1
postStimTime = 0.6
for probe in probeNames:
    for stim in stimNames:
        trials = trialStim==stim
        startTimes = vsyncTimes[stimStartFrame[trials]] + stimLatency[trials]
        for u in goodUnits[probe]:
            s,sdfTime = getSdf(unitData[probe][u]['times'],startTimes-preStimTime,preStimTime+postStimTime)
            sdfs[probe][stim].append(s)
    
for probe in probeNames:
    fig = plt.figure(figsize=(4.5,9.5))
    fig.suptitle('probe '+probe+' ('+str(len(goodUnits[probe]))+' good units)',fontsize=8)
    axs = fig.subplots(len(stimNames),1)
    ymin = ymax = 0
    for i,(ax,stim) in enumerate(zip(axs,stimNames)):
        meanSdf = np.mean(sdfs[probe][stim],axis=0)
        ax.plot(sdfTime-preStimTime,meanSdf,'k')
        ymin = min(ymin,meanSdf.min())
        ymax = max(ymax,meanSdf.max())
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=8)
        if i==len(stimNames)-1:
            ax.set_xlabel('time from stim onset (s)',fontsize=8)
        else:
            ax.set_xticklabels([])
        if i==0:
            ax.set_ylabel('spikes/s',fontsize=8)
        title = stim
        if 'sound' in stim:
            title += ': '+soundType[stim]+' '+str(soundParam[stim])
            title += ' log2(kHz)' if soundType[stim]=='log sweep' else ' Hz'
        ax.set_title(title,fontsize=8)
    for ax in axs:
        ax.set_xlim([-preStimTime,postStimTime])
        ax.set_ylim([1.02*ymin,1.02*ymax])
    plt.tight_layout()
        

# single trial raster
stim = 'sound2'
trialInd = 0
preStimTime = 0.5
postStimTime = 1
for probe in probeNames:
    fig = plt.figure(figsize=(9,9))
    fig.suptitle('probe '+probe,fontsize=8)
    ax = fig.add_subplot(1,1,1)
    trial = np.where(trialStim==stim)[0][trialInd]
    startTime = vsyncTimes[stimStartFrame[trial]] + stimLatency[trial]
    nUnits = len(goodUnits[probe])
    ax.add_patch(matplotlib.patches.Rectangle([0,0],width=trialSoundDur[trial],height=nUnits+1,facecolor='0.5',edgecolor=None,alpha=0.2,zorder=0))
    for i,u in enumerate(goodUnits[probe]):
        spikeTimes = unitData[probe][u]['times']
        ind = (spikeTimes>startTime-preStimTime) & (spikeTimes<startTime+postStimTime)
        ax.vlines(spikeTimes[ind]-startTime,i+0.5,i+1.5,colors='k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=8)
    ax.set_xlim([-preStimTime,postStimTime])
    ax.set_ylim([0.5,nUnits+0.5])
    ax.set_yticks([1,nUnits])
    ax.set_xlabel('time from stimu onset (s)',fontsize=8)
    ax.set_ylabel('unit',fontsize=8)
    title = stim+', trial '+str(trialInd+1)
    if 'sound' in stim:
        title += ': '+soundType[stim]+' '+str(soundParam[stim])
        title += ' log2(kHz)' if soundType[stim]=='log sweep' else ' Hz'
    ax.set_title(title,fontsize=8)
    plt.tight_layout()

















