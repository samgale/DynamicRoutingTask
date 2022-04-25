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
trialVisStimFrames = d['trialVisStimFrames'][:]
trialSoundDur = d['trialSoundDur'][:]
trialSoundArray = d['trialSoundArray'][:]
soundSampleRate = d['soundSampleRate'][()]

d.close()

for stim in np.unique(trialStim):
    print(stim,np.sum(trialStim==stim))


#
preTime = 0.1
postTime = 0.1
for stim in np.unique(trialStim):
    if 'sound' in stim:
        fig = plt.figure(figsize=(6,8))
        trials = trialStim==stim
        for i,trial in enumerate(np.where(trials)[0]):
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
            d = microphoneData[startSample:endSample]
            sound = trialSoundArray[trial]
            soundInterp = np.interp(t[(t>=0) & (t<=stimDur)],np.arange(sound.size)/soundSampleRate,sound)
            c = np.correlate(d,soundInterp,'valid')
            soundLatency = t[np.argmax(c)]
            ax = fig.add_subplot(trials.sum(),1,i+1)
            ax.plot(t,d,color='k')
            ax.plot(soundLatency,0,'o',mec='r',mfc='none')
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










