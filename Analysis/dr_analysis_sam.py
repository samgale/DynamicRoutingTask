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

f = fileIO.getFile(rootDir=baseDir,fileType='*.hdf5')

d = h5py.File(f)

trialEndFrame = d['trialEndFrame'][:]
nTrials = trialEndFrame.size
trialStartFrame = d['trialStartFrame'][:nTrials]
stimStartFrame = d['trialStimStartFrame'][:nTrials]
trialStim = d['trialStim'][nTrials]



# frame intervals
frameRate = 60
frameIntervals = d['frameIntervals'][:]
frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))
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
ax.set_ylabel('count')
plt.tight_layout()


# quiescent violations















d.close()



