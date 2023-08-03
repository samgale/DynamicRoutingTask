# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:40:28 2023

@author: svc_ccg
"""

import random
import TaskControl


class OptoTagging(TaskControl):
    
    def __init__(self,params,devName='laser_488',trialsPerType=30,power=[5],dur=[0.01,0.2],onRamp=0.001,offRamp=0.001,interval=60,intervalJitter=6):
        TaskControl.__init__(self,params)
        
        self.monBackgroundColor = -1
        self.maxFrames = 10 * 3600
        
        optoParamsPath = params['optoParamsPath'] if 'optoParamsPath' in params else None
        self.getOptoParams(optoParamsPath)
        
        self.optoDevName = devName
        self.trialsPerType = trialsPerType
        self.optoPower = power
        self.optoDur = dur
        self.optoOnRamp = onRamp
        self.optoOffRamp = offRamp
        self.optoInterval = interval
        self.optoIntervalJitter = intervalJitter
        
        from OptoParams import getBregmaGalvoCalibrationData, bregmaToGalvo, getOptoPowerCalibrationData, powerToVolts

        self.bregmaGalvoCalibrationData = getBregmaGalvoCalibrationData(self.rigName)
        self.optoPowerCalibrationData = getOptoPowerCalibrationData(self.rigName,self.optoDevName)
        self.optoOffsetVoltage = self.optoPowerCalibrationData['offsetV']

        with open(params['optoTaggingLocs'],'r') as f:
            cols = zip(*[line.strip('\n').split('\t') for line in f.readlines()]) 
        self.optoTaggingLocs = {}
        for d in cols:
            if d[0] == 'label':
                self.optoTaggingLocs[d[0]] = [s for s in d[1:]]
            else:
                self.optoTaggingLocs[d[0]] = [float(s) for s in d[1:]]
        
        self.optoBregma = [(x,y) for x,y in zip(self.optoTaggingLocs['X'],self.optoTaggingLocs['Y'])]
        self.galvoVoltage = [bregmaToGalvo(self.bregmaGalvoCalibrationData,x,y) for x,y in self.optoBregma]
        
        self.optoVoltage = [powerToVolts(self.optoPowerCalibrationData,pwr) for pwr in self.optoPower]
        
    def taskFlow(self):

        params = self.trialsPerType * list(itertools.product(self.optoDur,self.optoVoltage,self.galvoVoltage))
        random.shuffle(params)
        trial = -1
        interval = 5 * self.optoInterval
        
        self.trialOptoParamsIndex = []
        self.trialOptoDevice = []
        self.trialOptoOnsetFrame = []
        self.trialOptoDur = []
        self.trialOptoDelay = []
        self.trialOptoOnRamp = []
        self.trialOptoOffRamp = []
        self.trialOptoSinFreq = []
        self.trialOptoVoltage = []
        self.trialGalvoVoltage = []
        self.trialGavloDwellTime = []

        while self._continueSession:
            self.getInputData()
            
            if self._trialFrame == interval:
                if trial < len(params):
                    trial += 1
                    self._trialFrame = 0
                    interval = self.optoInterval + random.randint(0,self.optoIntervalJitter)
                    
                    dur,optoVoltage,galvoVoltage = params[trial]
                    
                    self.trialOptoOnsetFrame.append(self._sessionFrame)
                    self.trialOptoDur.append(dur)
                    self.trialOptoVoltage.append(optoVoltage)
                    self.trialGalvoVoltage.append(galvoVoltage)
                    
                    galvoX,galvoY = galvoVoltage
                    optoWaveform = self.getOptoPulseWaveform(amp=optoVoltage,
                                                             dur=dur,
                                                             onRamp=self.optoOnRamp,
                                                             offRamp=self.optoOffRamp,
                                                             offset=self.optoOffsetVoltage)
                    self._opto = [optoWaveform,galvoX,galvoY]
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