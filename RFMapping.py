# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:41:48 2019

@author: SVC_CCG
"""

from __future__ import division
import itertools
import random
import numpy as np
from psychopy import visual
from TaskControl import TaskControl


class RFMapping(TaskControl):
    
    def __init__(self,rigName,taskVersion=None):
        TaskControl.__init__(self,rigName)
        
        self.maxFrames = None
        self.maxBlocks = 6
        self.stimFrames = 15
        self.interStimFrames = 15
        
        # visual stimulus params
        # parameters that can vary across trials are lists
        self.warp = 'spherical'
        self.fullFieldContrast = [-1,0,1]
        self.gratingContrast = 1
        self.gratingSize = 20 # degrees
        self.gratingSF = 0.08 # cycles/deg
        self.gratingTF = 4 # cycles/s
        self.gratingOri = np.arange(0,360,45)
        self.gratingEdge= 'raisedCos'
        self.gratingEdgeBlurWidth = 0.08
        
        # auditory stimulus params
        self.soundDur = 0.25 # seconds
        self.soundVolume = 1 if rigName=='NP3' else 0.1
        self.noiseFreq = [2000,20000] # bandwidth, Hz
        self.toneFreq = np.arange(4000,16001,1000) # Hz
        self.saveSoundArray = True
        self.soundRandomSeed = 0
        
        if taskVersion is not None:
            self.setDefaultParams(taskVersion)
    
    
    def setDefaultParams(self,taskVersion):
        if True:
            pass
        else:
            raise ValueError(taskVersion + ' is not a recognized task version')
            
            
    def checkParamValues(self):
        pass
        

    def taskFlow(self):
        self.checkParamValues()
        
        # create visual stimulus
        gratingSizePix = self.gratingSize * self.pixelsPerDeg
        gratingStim = visual.GratingStim(win=self._win,
                                         units='pix',
                                         tex='sin',
                                         mask=self.gratingEdge,
                                         maskParams={'fringeWidth':self.gratingEdgeBlurWidth},
                                         size=gratingSizePix, 
                                         sf=self.gratingSF/self.pixelsPerDeg,
                                         contrast=self.gratingContrast)
            
        # calculate vis stim grid positions
        self.gridX,self.gridY = [np.linspace(-s/2 + gratingSizePix/2, s/2 - gratingSizePix/2, int(np.ceil(s/gratingSizePix))) for s in self.monSizePix]
          
        # make list of stimulus parameters for each trial
        n = len(self.gratingOri)//2 # non-vis trial multiplier
        trialParams = n * list(itertools.product(self.fullFieldContrast,[np.nan],[np.nan],[np.nan],[np.nan],[[np.nan,np.nan]]))
        trialParams += list(itertools.product([np.nan],self.gridX,self.gridY,self.gratingOri,[np.nan],[[np.nan,np.nan]]))
        trialParams += n * [(np.nan,)*5 + (self.noiseFreq,)]
        trialParams += n * list(itertools.product([np.nan],[np.nan],[np.nan],[np.nan],self.toneFreq,[[np.nan,np.nan]]))
        
        
        # things to keep track of
        self.stimStartFrame = []
        self.trialFullFieldContrast = []
        self.trialVisXY = []
        self.trialGratingOri = []
        self.trialToneFreq = []
        self.trialNoiseFreq = []
        self.trialSoundArray = []
        block = -1 # index of current block
        blockTrial = 0 # index of current trial in block
        
        # run loop for each frame presented on the monitor
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getInputData()
            
            # if starting new trial
            if self._trialFrame == 0:
                
                if block < 0 or blockTrial == len(trialParams):
                    block += 1
                    if self.maxBlocks is not None and block == self.maxBlocks:
                        break # end session
                    else:
                        # start new block of trials
                        blockTrial = 0
                        random.shuffle(trialParams)
                
                fullFieldContrast,x,y,ori,toneFreq,noiseFreq = trialParams[blockTrial]
                pos = (x,y)
                
                if not np.isnan(ori):
                    gratingStim.pos = pos
                    gratingStim.ori = ori
                    gratingStim.phase = 0
                
                if not np.isnan(toneFreq):
                    soundArray = self.makeSoundArray('tone',self.soundDur,self.soundVolume,toneFreq,None,self.soundRandomSeed)
                elif not all(np.isnan(noiseFreq)):
                    soundArray = self.makeSoundArray('noise',self.soundDur,self.soundVolume,noiseFreq,None,self.soundRandomSeed)
                else:
                    soundArray = np.array([])
                
                self.stimStartFrame.append(self._sessionFrame)
                self.trialFullFieldContrast.append(fullFieldContrast)
                self.trialVisXY.append(pos)
                self.trialGratingOri.append(ori)
                self.trialToneFreq.append(toneFreq)
                self.trialNoiseFreq.append(noiseFreq)
                if self.saveSoundArray:
                    self.trialSoundArray.append(soundArray)

            # show/trigger stimulus
            if self._trialFrame == 0 and soundArray.size > 0:
                if self.soundMode == 'internal':
                    self._sound = [soundArray]

            if not np.isnan(fullFieldContrast):
                if self._trialFrame ==0:
                    self._win.color = fullFieldContrast
                elif self._trialFrame == self.stimFrames:
                    self._win.color = self.monBackgroundColor
            
            if self._trialFrame < self.stimFrames and not np.isnan(ori):
                gratingStim.phase = gratingStim.phase + self.gratingTF/self.frameRate
                gratingStim.draw()
  
            # end trial after stimFrames and interStimFrames
            if self._trialFrame == self.stimFrames + self.interStimFrames:
                self._trialFrame = -1
                blockTrial += 1
            
            self.showFrame()



if __name__ == "__main__":
    import sys,json
    paramsPath = sys.argv[1]
    with open(paramsPath,'r') as f:
        params = json.load(f)
    task = RFMapping(params['rigName'],params['taskVersion'])
    task.start(params['subjectName'])