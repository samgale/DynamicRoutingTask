import random
import numpy as np
import matplotlib.pyplot as plt


def getBlockDurations(minDur,meanDur,maxDur):
    dur = maxDur + 1
    while dur > maxDur:
        dur = minDur + random.expovariate(1/(meanDur-minDur))
    return dur


blockDur = []
for session in range(int(1e4)):
    blockDur.append([])
    while sum(blockDur[-1]) < 3600:
        blockDur[-1].append(getBlockDurations(300,600,900))
        

allBlocks = [b for session in blockDur for b in session]       

blocksPerSession = [len(b) for b in blockDur]

sessionDur = [sum(b) for b in blockDur]

visDur = []
audDur = []
fracVis = []
for i,session in enumerate(blockDur):
    a = sum(session[0::2])
    b = sum(session[1::2])
    vis,aud = (a,b) if i % 2 else (b,a)
    visDur.append(vis)
    audDur.append(aud)
    fracVis.append(vis/(vis+aud))


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bins = np.arange(300,901,60)
h = np.histogram(allBlocks,bins=bins)[0]/len(allBlocks)
ax.bar(bins[:-1],h,width=60,color='k',align='edge',edgecolor='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Block duration (s)')
ax.set_ylabel('Probability')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bins = np.arange(3,13,1)+1
h = np.histogram(blocksPerSession,bins=bins)[0]/len(blocksPerSession)
ax.bar(bins[:-1],h,width=1,color='k',align='edge',edgecolor='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Blocks per session')
ax.set_ylabel('Probability')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bins = np.arange(3600,4500,60)
h = np.histogram(sessionDur,bins=bins)[0]/len(sessionDur)
ax.bar(bins[:-1],h,width=60,color='k',align='edge',edgecolor='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Session duration (s)')
ax.set_ylabel('Probability')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bins = np.arange(0,1,0.01)
h = np.histogram(fracVis,bins=bins)[0]/len(sessionDur)
ax.bar(bins[:-1],h,width=0.01,color='k',align='edge',edgecolor='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Fracion of session visual rewarded')
ax.set_ylabel('Probability')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fracVisSort = np.sort(fracVis)
cumProb = np.array([np.sum(fracVisSort<=i)/fracVisSort.size for i in fracVisSort])
ax.plot(fracVisSort,cumProb,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlabel('Fraction of session visual rewarded',fontsize=14)
ax.set_ylabel('Cumalative probability',fontsize=14)
plt.tight_layout()   
    




