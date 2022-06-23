# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:51:54 2022

@author: svc_ccg
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sync import sync
import probeSync
import ecephys



# behavior/stimuli
behavPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625820_06222022\DynamicRouting1_test_20220622_142520.hdf5"
d = h5py.File(behavPath,'r')

trialEndFrame = d['trialEndFrame'][:]
nTrials = trialEndFrame.size
trialStartFrame = d['trialStartFrame'][:nTrials]
stimStartFrame = d['trialStimStartFrame'][:nTrials]
trialStim = d['trialStim'].asstr()[:nTrials]
stimNames = np.unique(trialStim)
trialVisStimFrames = d['trialVisStimFrames'][:]
trialSoundDur = d['trialSoundDur'][:]
trialSoundArray = d['trialSoundArray'][:]
soundSampleRate = d['soundSampleRate'][()]

d.close()


# sync
syncPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625820_06222022\20220622T142413.h5"
syncDataset = sync.Dataset(syncPath)
    
vsyncRising,vsyncFalling = probeSync.get_sync_line_data(syncDataset,'vsync_stim')
vsyncTimes = vsyncFalling[1:] if vsyncFalling[0] < vsyncRising[0] else vsyncFalling

syncBarcodeRising,syncBarcodeFalling = probeSync.get_sync_line_data(syncDataset,'barcode_ephys')

syncBarcodeTimes,syncBarcodes = ecephys.extract_barcodes_from_times(syncBarcodeRising,syncBarcodeFalling)


# ephys sync data
ephysPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625820_06222022\2022-06-22_14-25-10\Record Node 103\experiment1\recording1"

probeNames = ['B','C']
probeDirNames = ['Neuropix-PXI-100.0','Neuropix-PXI-100.2']
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

stimLatency = np.full(nTrials,np.nan)
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
    else:
        stimLatency[trial] = 0


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





