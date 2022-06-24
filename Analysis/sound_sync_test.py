# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:51:54 2022

@author: svc_ccg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sync import sync
import probeSync
import ecephys


# file paths
syncPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Pilot ephys\sound_sync_test_04252022_1\20220425T102323.h5"
datPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Pilot ephys\sound_sync_test_04252022_1\2022-04-25_10-23-29\Record Node 105\experiment1\recording1\continuous\NI-DAQmx-103.0\continuous.dat"
ttlStatesPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Pilot ephys\sound_sync_test_04252022_1\2022-04-25_10-23-29\Record Node 105\experiment1\recording1\events\NI-DAQmx-103.0\TTL_1\channel_states.npy"
behavPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Pilot ephys\sound_sync_test_04252022_1\DynamicRouting1_test_20220425_102338.hdf5"

ttlTimestampsPath = os.path.join(os.path.dirname(ttlStatesPath),'timestamps.npy')
datTimestampsPath = os.path.join(os.path.dirname(datPath),'timestamps.npy')


# sync
syncDataset = sync.Dataset(syncPath)
    
vsyncRising,vsyncFalling = probeSync.get_sync_line_data(syncDataset,'vsync_stim')
vsyncTimes = vsyncFalling[1:] if vsyncFalling[0] < vsyncRising[0] else vsyncFalling

syncBarcodeRising,syncBarcodeFalling = probeSync.get_sync_line_data(syncDataset,'barcode_ephys')

syncBarcodeTimes,syncBarcodes = ecephys.extract_barcodes_from_times(syncBarcodeRising,syncBarcodeFalling)


# ephys
ephysSampleRate = 30000

numAnalogCh = 8
datData = np.memmap(datPath,dtype='int16',mode='r')    
datData = np.reshape(datData,(int(datData.size/numAnalogCh),-1)).T

datTimestamps = np.load(datTimestampsPath)

speakerCh = 1
microphoneCh = 3
speakerData = datData[speakerCh]
microphoneData = datData[microphoneCh]

ttlStates = np.load(ttlStatesPath)
ttlTimestamps = np.load(ttlTimestampsPath) - datTimestamps[0]

ephysBarcodeRising = ttlTimestamps[ttlStates>0]/ephysSampleRate
ephysBarcodeFalling = ttlTimestamps[ttlStates<0]/ephysSampleRate
ephysBarcodeTimes,ephysBarcodes = ecephys.extract_barcodes_from_times(ephysBarcodeRising,ephysBarcodeFalling)

ephysShift,relSampleRate,endpoints = ecephys.get_probe_time_offset(syncBarcodeTimes,syncBarcodes,ephysBarcodeTimes,ephysBarcodes,0,ephysSampleRate)


# behavior/stimuli
d = h5py.File(behavPath,'r')

trialEndFrame = d['trialEndFrame'][:]
nTrials = trialEndFrame.size
trialStartFrame = d['trialStartFrame'][:nTrials]
stimStartFrame = d['trialStimStartFrame'][:nTrials]
trialStim = d['trialStim'][:nTrials]
trialVisStimFrames = d['trialVisStimFrames'][:nTrials]
trialSoundDur = d['trialSoundDur'][:nTrials]
trialSoundArray = d['trialSoundArray'][:nTrials]
soundSampleRate = d['soundSampleRate'][()]

d.close()

stimNames = np.unique(trialStim)
for stim in stimNames:
    print(stim,np.sum(trialStim==stim))


#
signalNames = ('speaker','microphone')
soundLatency = {stim: {sig:[] for sig in signalNames} for stim in stimNames if 'sound' in stim}
preTime = 0.1
postTime = 0.1
for stim in stimNames:
    if 'sound' in stim:
        fig = plt.figure(figsize=(6,8))
        trials = trialStim==stim
        for i,trial in enumerate(np.where(trials)[0]):
            ax = fig.add_subplot(trials.sum(),1,i+1)
            startFrame = stimStartFrame[trial]
            startTime = vsyncTimes[startFrame]
            if 'vis' in stim:
                stimDur = vsyncTimes[startFrame+trialVisStimFrames[trial]] - startTime
            elif 'sound' in stim:
                stimDur = trialSoundDur[trial]
            else:
                stimDur = 0.5
            startSample = int((startTime+ephysShift-preTime)*relSampleRate)
            endSample = int((startTime+ephysShift+stimDur+postTime)*relSampleRate)
            t = np.arange(endSample-startSample)/relSampleRate-preTime
            sound = trialSoundArray[trial]
            tInterp = np.arange(-preTime,stimDur+postTime,1/soundSampleRate)
            for d,sig,marker,alpha in zip((speakerData,microphoneData),signalNames,('o','x'),(1,0.5)):
                d = d[startSample:endSample]
                dInterp = np.interp(tInterp,t,d)
                c = np.correlate(dInterp,sound,'valid')
                soundLatency[stim][sig].append(tInterp[np.argmax(c)])
                ax.plot(t,d,color='k',alpha=alpha)
                ax.plot(soundLatency[stim][sig][-1],0,marker,mec='r',mfc='none')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([0,0.05])
            if i==0:
                ax.set_title(stim)
            if i==trials.sum()-1:
                ax.set_xlabel('time from stim start (s)')
            else:
                ax.set_xticklabels([])
        plt.tight_layout()


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(2,1,1)
ymax = 0
for stim in soundLatency:
    x = int(stim[5:])
    for sig,marker in zip(signalNames,('o','x')):
        y = soundLatency[stim][sig]
        lbl = sig if x==1 else None
        ax.plot(x+np.zeros(len(y)),y,marker,mec='k',mfc='none',alpha=0.25,label=lbl)
        ax.plot(x,np.median(y),marker,mec='r',mfc='none')
        ymax = max(ymax,max(y))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,len(soundLatency)+1])
ax.set_ylim([0,1.05*ymax])
ax.set_xlabel('sound')
ax.set_ylabel('sound latency (s)')
ax.legend()

ax = fig.add_subplot(2,1,2)
amin = amax = 0
ax.plot([-1,1],[-1,1],'--',color='0.75')
for stim in soundLatency:
    x,y = [soundLatency[stim][sig] for sig in signalNames]
    ax.plot(x,y,'o',mec='k',mfc='none')
    amin = min(amin,min(x),min(y))
    amax = max(amax,max(x),max(y))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
alim = [amin-0.05*amax,amax+0.05*amax]
ax.set_xlim(alim)
ax.set_ylim(alim)
ax.set_aspect('equal')
ax.set_xlabel('speaker latency (s)')
ax.set_ylabel('microphone latency (s)')

plt.tight_layout()





