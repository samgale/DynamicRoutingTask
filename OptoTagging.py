# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:40:28 2023

@author: svc_ccg
"""

import itertools
import random
import TaskControl
from OptoParams import getBregmaGalvoCalibrationData, bregmaToGalvo, getOptoPowerCalibrationData, powerToVolts


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
        params = {d[0]: d[1:] for d in cols}
        self.optoTaggingLocs = {key: [val for val,useVal in zip(vals,params['use']) if useVal in ('True','true')] for key,vals in params.items() if key != 'use'}
        for key,vals in self.optoParams.items():
            if key == 'label':
                pass
            elif key == 'device':
                self.optoTaggingLocs[key] = [val.split(',') for val in vals]
            else:
                self.optoTaggingLocs[key] = [float(val) for val in vals]
        
        self.bregmaXY = [(x,y) for x,y in zip(self.optoTaggingLocs['bregmaX'],self.optoTaggingLocs['bregmaY'])]
        self.bregmaGalvoCalibrationData = getBregmaGalvoCalibrationData(self.rigName)
        self.galvoVoltage = [bregmaToGalvo(self.bregmaGalvoCalibrationData,x,y) for x,y in self.bregmaXY]
        
        devNames = set(d for dev in self.optoTaggingLocs['device'] for d in dev)
        assert(len(devNames) == 1)
        self.optoDev = list(devNames)[0]
        self.optoPowerCalibrationData = getOptoPowerCalibrationData(self.rigName,self.optoDev)
        self.optoOffsetVoltage = self.optoPowerCalibrationData['offsetV']
        self.optoVoltage = [powerToVolts(self.optoPowerCalibrationData,pwr) for pwr in self.optoPower]
        
        
    def taskFlow(self):

        params = self.trialsPerType * list(itertools.product(self.optoDur,self.optoVoltage,list(zip(self.optoTaggingLocs['label'],self.galvoVoltage))))
        random.shuffle(params)
        trial = -1
        interval = 5 * self.optoInterval
        
        self.trialOptoLabel = []
        self.trialOptoDur = []
        self.trialOptoVoltage = []
        self.trialGalvoVoltage = []

        while self._continueSession:
            self.getInputData()
            
            if self._trialFrame == interval:
                if trial < len(params):
                    trial += 1
                    self._trialFrame = 0
                    interval = self.optoInterval + random.randint(0,self.optoIntervalJitter)
                    
                    dur,optoVoltage,(optoLabel,galvoVoltage) = params[trial]
                    
                    self.trialOptoLabel.append(optoLabel)
                    self.trialOptoDur.append(dur)
                    self.trialOptoVoltage.append(optoVoltage)
                    self.trialGalvoVoltage.append(galvoVoltage)
                    
                    optoWaveform = [self.getOptoPulseWaveform(amp=optoVoltage,dur=dur,onRamp=self.optoOnRamp,offRamp=self.optoOffRamp,offset=self.optoOffsetVoltage)]

                    galvoX,galvoY = galvoVoltage
                    
                    self._opto = [[self.optoDev],optoWaveform,galvoX,galvoY]
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