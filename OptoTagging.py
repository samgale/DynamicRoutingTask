# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:40:28 2023

@author: svc_ccg
"""

import itertools
import random
from TaskControl import TaskControl
import TaskUtils


class OptoTagging(TaskControl):
    
    def __init__(self,params):
        TaskControl.__init__(self,params)
        
        self.monBackgroundColor = -1
        self.maxFrames = 10 * 3600
        
        self.trialsPerType = 25
        self.optoPower = [5]
        self.optoDur = [0.01,0.2]
        self.optoOnRamp = 0.001
        self.optoOffRamp = 0.001
        self.optoInterval = 60
        self.optoIntervalJitter = 6
        
        
        with open(params['optoTaggingLocs'],'r') as f:
            cols = zip(*[line.strip('\n').split('\t') for line in f.readlines()])
        self.optoTaggingLocs = {d[0]: d[1:] for d in cols}
        for key,vals in self.optoTaggingLocs.items():
            if key == 'label':
                pass
            elif key == 'device':
                self.optoTaggingLocs[key] = [val.split(',') for val in vals]
            else:
                self.optoTaggingLocs[key] = [float(val) for val in vals]
        
        self.bregmaXY = [(x,y) for x,y in zip(self.optoTaggingLocs['bregmaX'],self.optoTaggingLocs['bregmaY'])]
        self.bregmaGalvoCalibrationData = TaskUtils.getBregmaGalvoCalibrationData(self.rigName)
        self.galvoVoltage = [TaskUtils.bregmaToGalvo(self.bregmaGalvoCalibrationData,x,y) for x,y in self.bregmaXY]
        
        devNames = set(d for dev in self.optoTaggingLocs['device'] for d in dev)
        assert(len(devNames) == 1)
        self.optoDev = list(devNames)[0]
        self.optoPowerCalibrationData = TaskUtils.getOptoPowerCalibrationData(self.rigName,self.optoDev)
        self.optoOffsetVoltage = self.optoPowerCalibrationData['offsetV']
        self.optoVoltage = [TaskUtils.powerToVolts(self.optoPowerCalibrationData,pwr) for pwr in self.optoPower]
        
        
    def taskFlow(self):

        params = list(itertools.product(self.optoDur,self.optoVoltage,list(zip(self.optoTaggingLocs['label'],self.galvoVoltage))))
        
        trial = 0
        interval = self.optoInterval
        
        self.trialOptoOnsetFrame = []
        self.trialOptoLabel = []
        self.trialOptoDur = []
        self.trialOptoVoltage = []
        self.trialGalvoVoltage = []

        while self._continueSession:
            self.getInputData()
            
            if self._trialFrame == interval:
                if trial < len(params) * self.trialsPerType:
                    self._trialFrame = 0
                    
                    paramsIndex = trial % len(params)
                    if paramsIndex == 0:
                        random.shuffle(params)
                    dur,optoVoltage,(optoLabel,galvoVoltage) = params[paramsIndex]
                    
                    self.trialOptoOnsetFrame.append(self._sessionFrame)
                    self.trialOptoLabel.append(optoLabel)
                    self.trialOptoDur.append(dur)
                    self.trialOptoVoltage.append(optoVoltage)
                    self.trialGalvoVoltage.append(galvoVoltage)
                    
                    optoWaveform = [TaskUtils.getOptoPulseWaveform(self.optoSampleRate,amp=optoVoltage,dur=dur,onRamp=self.optoOnRamp,offRamp=self.optoOffRamp,offset=self.optoOffsetVoltage)]

                    galvoX,galvoY = galvoVoltage
                    
                    self.loadOptoWaveform([self.optoDev],optoWaveform,galvoX,galvoY)

                    self._opto = True

                    trial += 1
                    interval = self.optoInterval + random.randint(0,self.optoIntervalJitter)
                else:
                    self._continueSession = False

            self.showFrame()


if __name__ == "__main__":
    import sys,json
    paramsPath = sys.argv[1]
    with open(paramsPath,'r') as f:
        params = json.load(f)
    task = OptoTagging(params)
    task.start(params['subjectName'])