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
        
        self.monBackgroundColor = float(params['monBackgroundColor']) if 'monBackgroundColor' in params and params['monBackgroundColor'] is not None else -0.95
        self.maxFrames = int(params['maxFrames']) if 'maxFrames' in params and params['maxFrames'] is not None else None
        self.maxTrials = int(params['maxTrials']) if 'maxTrials' in params and params['maxTrials'] is not None else None
        
        self.trialsPerType = int(params['trialsPerType']) if 'trialsPerType' in params and params['trialsPerType'] is not None else 25
        self.optoPower = [float(pwr) for pwr in params['optoAmp'].strip('[],').split(',')] if 'optoAmp' in params and params['optoAmp'] is not None else [5] # mW
        self.optoDur = [0.01,0.2] # seconds
        self.optoOnRamp = 0.001 # seconds
        self.optoOffRamp = 0.001 # seconds
        self.optoInterval = 60 # frames
        self.optoIntervalJitter = 6 # max random frames added to interval
        
        if params is not None and 'taskVersion' in params and params['taskVersion'] is not None:
            self.taskVersion = params['taskVersion']
            self.setDefaultParams(params['taskVersion'])
        else:
            self.taskVersion = None
        
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
        self.bregmaOffsetXY = [(x,y) for x,y in zip(self.optoTaggingLocs['bregma offset X'],self.optoTaggingLocs['bregma offset Y'])]
        self.bregmaGalvoCalibrationData = TaskUtils.getBregmaGalvoCalibrationData(self.rigName)
        self.galvoVoltage = [TaskUtils.bregmaToGalvo(self.bregmaGalvoCalibrationData,x,y,offsetX,offsetY) for (x,y),(offsetX,offsetY) in zip(self.bregmaXY,self.bregmaOffsetXY)]
        
        devNames = set(d for dev in self.optoTaggingLocs['device'] for d in dev)
        self.optoPowerCalibrationData = {dev: TaskUtils.getOptoPowerCalibrationData(self.rigName,dev) for dev in devNames}
        self.optoOffsetVoltage = {dev: self.optoPowerCalibrationData[dev]['offsetV'] for dev in devNames}
        self.optoVoltage = {dev: {str(pwr): TaskUtils.powerToVolts(self.optoPowerCalibrationData[dev],pwr) for pwr in self.optoPower} for dev in devNames}
        
    
    def setDefaultParams(self,taskVersion):
        if True:
            pass
        else:
            raise ValueError(taskVersion + ' is not a recognized task version')
    
        
    def taskFlow(self):

        params = list(itertools.product(self.optoDur,self.optoPower,list(zip(self.optoTaggingLocs['label'],self.optoTaggingLocs['device'],self.galvoVoltage))))
        
        trial = 0
        interval = self.optoInterval
        
        self.trialOptoOnsetFrame = []
        self.trialOptoLabel = []
        self.trialOptoDevice = []
        self.trialOptoDur = []
        self.trialOptoPower = []
        self.trialGalvoVoltage = []

        while self._continueSession:
            self.getInputData()
            
            if self._trialFrame == interval:
                if trial < len(params) * self.trialsPerType and (self.maxTrials is None or trial < self.maxTrials):
                    self._trialFrame = 0
                    
                    paramsIndex = trial % len(params)
                    if paramsIndex == 0:
                        random.shuffle(params)
                    dur,pwr,(optoLabel,optoDevice,galvoVoltage) = params[paramsIndex]
                    
                    self.trialOptoOnsetFrame.append(self._sessionFrame)
                    self.trialOptoLabel.append(optoLabel)
                    self.trialOptoDevice.append(optoDevice)
                    self.trialOptoDur.append(dur)
                    self.trialOptoPower.append(pwr)
                    self.trialGalvoVoltage.append(galvoVoltage)
                    
                    optoWaveform = [TaskUtils.getOptoPulseWaveform(self.optoSampleRate,amp=self.optoVoltage[dev][str(pwr)],dur=dur,onRamp=self.optoOnRamp,offRamp=self.optoOffRamp,offset=self.optoOffsetVoltage[dev]) for dev in optoDevice]

                    galvoX,galvoY = galvoVoltage
                    
                    self.loadOptoWaveform(optoDevice,optoWaveform,galvoX,galvoY)

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