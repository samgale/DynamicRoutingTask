# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:51:54 2022

@author: svc_ccg
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from sync import sync
import probeSync
import ecephys


# sync
syncPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Pilot ephys\04142022_test\20220414T10284.h5"

syncDataset = sync.Dataset(syncPath)
    
vsyncRising,vsyncFalling = probeSync.get_sync_line_data(syncDataset,'vsync_stim')
vsyncTimes = vsyncFalling[1:] if vsyncFalling[0] < vsyncRising[0] else vsyncFalling

syncBarcodeRising,syncBarcodeFalling = probeSync.get_sync_line_data(syncDataset,'barcode_ephys')

syncBarcodeTimes,syncBarcodes = ecephys.extract_barcodes_from_times(syncBarcodeRising,syncBarcodeFalling)

# ephys
datPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Pilot ephys\04142022_test\experiment3\recording1\continuous\NI-DAQmx-103.0\continuous.dat"

sampleRate = 30000

numAnalogCh = 8
datData = np.memmap(datPath,dtype='int16',mode='r')    
datData = np.reshape(datData,(int(datData.size/numAnalogCh),-1)).T

speakerCh = 1
microphoneCh = 3
speakerData = datData[speakerCh]
microphoneData = datData[microphoneCh]

ttlStatesPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Pilot ephys\04142022_test\experiment3\recording1\events\NI-DAQmx-103.0\TTL_1\channel_states.npy"
ttlTimestampsPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Pilot ephys\04142022_test\experiment3\recording1\events\NI-DAQmx-103.0\TTL_1\timestamps.npy"

ttlStates = np.load(ttlStatesPath)
ttlTimestamps = np.load(ttlTimestampsPath)

ephysBarcodeRising = ttlTimestamps[ttlStates>0]/sampleRate
ephysBarcodeFalling = ttlTimestamps[ttlStates<0]/sampleRate
ephysBarcodeTimes,ephysBarcodes = ecephys.extract_barcodes_from_times(ephysBarcodeRising,ephysBarcodeFalling)

shift,relSampleRate,endpoints = ecephys.get_probe_time_offset(syncBarcodeTimes,syncBarcodes,ephysBarcodeTimes,ephysBarcodes,0,sampleRate)


# behavior/stimuli
behavPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Pilot ephys\04142022_test\DynamicRouting1_test_20220414_102817.hdf5"

d = h5py.File(behavPath,'r')

trialEndFrame = d['trialEndFrame'][:]
nTrials = trialEndFrame.size
trialStartFrame = d['trialStartFrame'][:nTrials]
stimStartFrame = d['trialStimStartFrame'][:nTrials]
trialStim = d['trialStim'][:nTrials]
trialVisStimFrames = d['trialVisStimFrames'][:]
trialSoundDur = d['trialSoundDur'][:]

d.close()


#
preTime = 0.5
postTime = 0.5
for stim in np.unique(trialStim):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for trial in np.where(trialStim==stim)[0]:
        startFrame = stimStartFrame[trial]
        startTime = vsyncTimes[startFrame]
        if 'vis' in stim:
            stimDur = vsyncTimes[startFrame+trialVisStimFrames[trial]] - startTime
        elif 'sound' in stim:
            stimDur = trialSoundDur[trial]
        else:
            stimDur = 0.5
        startSample = int((startTime-shift-preTime)*relSampleRate)
        endSample = int((startTime-shift+stimDur+postTime)*relSampleRate)
        t = np.arange(endSample-startSample)/relSampleRate-preTime
        ax.plot(t,microphoneData[startSample:endSample],'k')
        break
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('time from stim start (s)')
    ax.set_title(stim)
    plt.tight_layout()










