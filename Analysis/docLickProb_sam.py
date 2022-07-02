# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:27:03 2022

@author: svc_ccg
"""

import random
import numpy as np
import matplotlib.pyplot as plt

    
class DocSim():
    
    def __init__(self,nTrials,lickProbChange=0,lickProbTiming=[0],timingProb=0):
        self.nTrials = nTrials
        self.lickProbChange = lickProbChange
        self.lickProbTiming = lickProbTiming
        self.timingProb = timingProb
        self.catchProb = 0.125
        self.timeoutFlashes = 0
        self.graceFlashes = 5
        self.maxAborts = 5
    
    def pickChangeFlash(self,p=0.3,nmin=5,nmax=12):
        n = np.random.geometric(p)
        if nmin <= n <= nmax:
            return n
        else:
            return self.pickChangeFlash()
    
    def runTrial(self):
        self.trialStartFlash.append(self.flash+1)
        if self.aborts <= self.maxAborts:
            self.trialChangeFlash = self.pickChangeFlash()
        lick = False
        outcome = False
        for trialFlash in range(1,self.trialChangeFlash+1):
            self.flash += 1
            if trialFlash == self.trialChangeFlash:
                if random.random() < self.catchProb:
                    isCatch,isChange = True,False
                    self.catchFlash.append(self.flash)
                else:
                    isCatch,isChange = False,True
                    self.changeFlash.append(self.flash)
            else:
                isCatch = isChange = False
            lick = False
            if random.random() < self.timingProb:
                if trialFlash <= len(self.lickProbTiming) and random.random() < self.lickProbTiming[trialFlash-1]:
                    lick = True
            elif isChange and random.random() < self.lickProbChange:
                lick = True
            if isChange:
                outcome = 'hit' if lick else 'miss'
            elif isCatch:
                outcome = 'false alarm' if lick else 'correct reject'
            elif lick:
                outcome = 'abort'
            if outcome:
                if outcome == 'abort':
                    if self.aborts < self.maxAborts:
                        self.aborts += 1
                    else:
                        self.aborts = 0
                    self.flash += self.timeoutFlashes
                else:
                    self.aborts = 0
                    self.flash += self.graceFlashes
                self.trialOutcome.append(outcome)
                break
                
    def runSession(self):
        self.flash = 0
        self.trialStartFlash = []
        self.changeFlash = []
        self.catchFlash = []
        self.trialOutcome = []
        self.aborts = 0
        for _ in range(self.nTrials):
            self.runTrial()
        



    

# n = np.array([pickChangeFlash(p,nmin,nmax) for _ in range(100000)])
# flashNum = np.arange(1,n.max()+1)
# changeProb = np.array([np.sum(n==i)/(n.size/(1-catchProb)) for i in flashNum])
# conditionalChangeProb = np.array([changeProb[i]/(changeProb[i:].sum()+catchProb) for i in range(n.max())])


n = np.array([getChangeFlash() for _ in range(100000)])

flashNum = np.arange(1,n.max()+1)

changeProb = np.array([np.sum(n==i)/(n.size/(1-catchProb)) for i in flashNum])

conditionalChangeProb = np.array([changeProb[i]/(changeProb[i:].sum()+catchProb) for i in range(n.max())])

abortProb = np.zeros(flashNum[-1])
for i in range(1,flashNum[-2]):
    abortProb[i+1] = (1-abortProb[i-1])*conditionalChangeProb[i]

changeProbWithAborts = changeProb * (1-abortProb)

predictedLickProb = changeProbWithAborts * conditionalChangeProb

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,flashNum[-1]],[catchProb]*2,'--',color='0.5',label='catch prob.')
ax.plot(flashNum,changeProb,'ko-',label='change prob. (including catch trials)')
ax.plot(flashNum,conditionalChangeProb,'bo-',label='change prob. given no change previous flash')
ax.plot(flashNum,changeProbWithAborts,'ro-',label='change prob with predicted aborts')
ax.plot(flashNum,predictedLickProb,'g-',label='predicted lick prob. (blue * red)')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flash number')
ax.set_ylabel('probability')
ax.legend(bbox_to_anchor=(0.5,1),fontsize=8)
plt.tight_layout()




