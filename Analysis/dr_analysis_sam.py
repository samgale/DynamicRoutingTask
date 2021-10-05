# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:55:44 2021

@author: svc_ccg
"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
import fileIO


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\Sam"

f = fileIO.getFile(rootDir=baseDir,fileType='*.hdf5','r')

d = h5py.File(f)


#
frameRate = 60
frameIntervals = d['frameIntervals'][:]
frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))

trialEndFrame = d['trialEndFrame'][:]
nTrials = trialEndFrame.size
trialStartFrame = d['trialStartFrame'][:nTrials]
stimStartFrame = d['trialStimStartFrame'][:nTrials]
stimStartTimes = frameTimes[stimStartFrame]

trialStim = d['trialStim'][:nTrials]
rewardedStim = d['blockStimRewarded'][:nTrials][d['trialBlock'][:nTrials]-1]

trialResponse = d['trialResponse'][:nTrials]
autoRewarded = d['trialAutoRewarded'][:nTrials]
rewardEarned = d['trialRewarded'][:nTrials] & (~autoRewarded)

catchTrials = trialStim == 'catch'
goTrials = (trialStim == rewardedStim) & (~autoRewarded)
nogoTrials = (trialStim != rewardedStim) & (~catchTrials)

assert(nTrials == goTrials.sum() + nogoTrials.sum() + autoRewarded.sum() + catchTrials.sum())

hitTrials = goTrials & responseTrials
missTrials = goTrials & (~responseTrials)
falseAlarmTrials = nogoTrials & responseTrials
correctRejectTrials = nogoTrials & (~responseTrials)

hitRate = hitTrials.sum() / hitTrials.size
falseAlarmRate = falseAlarmTrials.sum() / falseAlarmTrials.size
catchRate = catchTrials.sum() / catchTrials.size


# frame intervals
longFrames = frameIntervals > 1.5/frameRate

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bins = np.arange(-0.5/frameRate,frameIntervals.max()+1/frameRate,1/frameRate)
ax.hist(frameIntervals,bins=bins,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_yscale('log')
ax.set_xlabel('frame interval (s)')
ax.set_ylabel('count')
ax.set_title(str(round(100 * longFrames.sum() / longFrames.size,2)) + '% of frames long')
plt.tight_layout()


# inter-trial intervals
interTrialIntervals = np.diff(frameTimes[stimStartFrame])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(interTrialIntervals,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('inter-trial interval (s)')
ax.set_ylabel('trials')
plt.tight_layout()


# quiescent violations
quiescentViolationFrames = d['quiescentViolationFrames'][:]
trialQuiescentViolations = []
for sf,ef in zip(trialStartFrame,trialEndFrame):
    trialQuiescentViolations.append(np.sum((quiescentViolationFrames > sf) & (quiescentViolationFrames < ef)))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(frameTimes[quiescentViolationFrames],np.arange(quiescentViolationFrames.size)+1,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('time (s)')
ax.set_ylabel('quiescent period violations')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bins = np.arange(-0.5,max(trialQuiescentViolations)+1,1)
ax.hist(trialQuiescentViolations,bins=bins,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('quiescent violations per trial (s)')
ax.set_ylabel('trials')
plt.tight_layout()


# lick raster
lickFrames = d['lickFrames'][:]
lickTimes = frameTimes[lickFrames]

preTime = 4
postTime = 4
for trials,trialType in zip((np.ones(nTrials,dtype=bool),goTrials,nogoTrials,autoRewarded,catchTrials),
                            ('all','go','no-go','auto reward','catch')):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    respTrials = trialResponse[trials]
    for i,st in enumerate(stimStartTimes[trials]):
        lt = lickTimes - st
        trialLickTimes = lt[(lt >= -preTime) & (lt <= postTime)]
        clr = 'm' if respTrials[i] else 'k'
        ax.vlines(trialLickTimes,i-0.5,i+0.5,colors=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel('trial')
    ax.set_title(trialType)
    plt.tight_layout()





# clean up
d.close()

