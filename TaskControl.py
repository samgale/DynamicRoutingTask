# -*- coding: utf-8 -*-
"""
Superclass for behavioral task control

"""

from __future__ import division
import math, os, time
from threading import Timer
import h5py
import numpy as np
from psychopy import monitors, visual, event
from psychopy.visual.windowwarp import Warper
import sounddevice
import nidaqmx


class TaskControl():
    
    def __init__(self,rigName):
        assert(rigName in ('NP3','E1','E2','E3','E4','E5','E6'))
        self.rigName = rigName
        self.subjectName = None
        self.maxFrames = None # max number of frames before task terminates
        self.saveParams = True # if True, saves all attributes not starting with underscore
        self.saveFrameIntervals = True
        self.monBackgroundColor = 0 # gray; can adjust this for luminance measurement
        self.minWheelAngleChange = 0 # radians per frame
        self.maxWheelAngleChange = 0.5 # radians per frame
        self.spacebarRewardsEnabled = True
        self.soundMode = 'internal' # internal (sound card) or external (nidaq digital trigger)
        self.soundSampleRate = 48000
        
        # rig specific settings
        self.saveDir= r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data"
        self.screen = 0 # monitor to present stimuli on
        self.monWidth = 52.0 # cm
        self.monDistance = 15.3 # cm
        self.monGamma = 2.3 # float or None
        self.monSizePix = (1920,1200)
        self.warp = None # 'spherical', 'cylindrical', 'warpfile', None
        self.warpFile = None
        self.wheelRadius = 8.25 # cm
        self.wheelPolarity = -1
        self.rotaryEncoderCh = 1
        self.digitalSolenoidTrigger = True
        self.solenoidOpenTime = 0.05 # seconds
        if self.rigName=='NP3':
            self.drawDiodeBox = True
            self.diodeBoxSize = 50
            self.diodeBoxPosition = (935,550)
            self.nidaqDevices = ('USB-6001',)
            self.nidaqDeviceNames = ('Dev0',)
        elif 'E' in rigName:
            self.drawDiodeBox = False
            self.nidaqDevices = ('USB-6009',)
            self.nidaqDeviceNames = ('Dev1',)
            

    def prepareSession(self):
        self._win = None
        self._nidaqTasks = []
        
        startTime = time.localtime()
        self.startTime = time.strftime('%Y%m%d_%H%M%S',startTime)
        print('start time was: ' + time.strftime('%I:%M',startTime))
        
        self.pixelsPerDeg = 0.5 * self.monSizePix[0] / math.degrees(math.atan(0.5 * self.monWidth / self.monDistance))
        
        self.prepareWindow()
        
        self.startNidaqDevice()
        
        self._keys = [] # list of keys pressed since previous frame
        
        self.rotaryEncoderVolts = [] # rotary endoder output at each frame
        self.wheelPosRadians = [] # absolute angle of wheel in radians
        self.deltaWheelPos = [] # angular change in wheel position
        self.lickFrames = []
        
        self._continueSession = True
        self._sessionFrame = 0 # index of frame since start of session
        self._trialFrame = 0 # index of frame since start of trial
        self._reward = False # reward delivered at next frame flip if True
        self.rewardFrames = [] # index of frames at which reward delivered
        self.manualRewardFrames = [] # index of frames at which reward manually delivered
        self.rewardSize = [] # size (solenoid open time) of each reward
        self._sound = False # sound triggered at next frame flip if True
        self._opto = False # False or dictionary of params for optoPulse at next frame flip
        
    
    def prepareWindow(self):
        self._mon = monitors.Monitor('monitor1',
                                     width=self.monWidth,
                                     distance=self.monDistance,
                                     gamma=self.monGamma)
        self._mon.setSizePix(self.monSizePix)
        self._mon.saveMon()
        self._win = visual.Window(monitor=self._mon,
                                  screen=self.screen,
                                  fullscr=True,
                                  units='pix',
                                  color=self.monBackgroundColor)
        self._warper = Warper(self._win,warp=self.warp,warpfile=self.warpFile)
        for _ in range(10):
            self._win.flip()
        self._win.setRecordFrameIntervals(self.saveFrameIntervals)
        
        if self.drawDiodeBox:
            self._diodeBox = visual.Rect(self._win,
                                         units='pix',
                                         width=self.diodeBoxSize,
                                         height=self.diodeBoxSize,
                                         lineColor=0,
                                         fillColor=1, 
                                         pos=self.diodeBoxPosition)
        
        
    def start(self,subjectName=None):
        try:
            if subjectName is not None:
                self.subjectName = str(subjectName)
            
            self.prepareSession()
            
            self.taskFlow()
        
        except:
            raise
            
        finally:
            self.completeSession()
    
    
    def taskFlow(self):
        # override this method in subclass
        
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getNidaqData()
            
            # do stuff, for example:
            # check for licks and/or wheel movement
            # update/draw stimuli
            
            self.showFrame()
    
    
    def showFrame(self):
        self._frameSignalOutput.write(True)
        
        # spacebar delivers reward
        # escape key ends session
        self._keys = event.getKeys()
        
        if self.spacebarRewardsEnabled and 'space' in self._keys and not self._reward:
            self._reward = self.solenoidOpenTime
            self.manualRewardFrames.append(self._sessionFrame)
        
        if 'escape' in self._keys or (self.maxFrames is not None and self._sessionFrame == self.maxFrames - 1):   
            self._continueSession = False
        
        if self._sound:
            if self.soundMode == 'internal':
                sounddevice.play(*self._sound)
            else:
                getattr(self,'_'+self._sound+'Output').write(True)
        
        # show new frame
        if self.drawDiodeBox:
            self._diodeBox.fillColor = -self._diodeBox.fillColor
            self._diodeBox.draw()
        self._win.flip()
        
        self._frameSignalOutput.write(False)
        
        if self._opto:
            self.optoPulse(**self._opto)
            self._opto = False
        
        if self._reward:
            self.triggerReward(self._reward)
            self.rewardFrames.append(self._sessionFrame)
            self.rewardSize.append(self._reward)
            self._reward = False
        
        if self._sound:
            if self.soundMode == 'external':
                getattr(self,'_'+self._sound+'Output').write(False)
            self._sound = False
            
        self._sessionFrame += 1
        self._trialFrame += 1
                                               
    
    def completeSession(self):
        try:
            if self._win is not None:
                self._win.close()
            self.stopNidaqDevice()
        except:
            raise
        finally:
            if self.saveParams:
                if self.subjectName is None:
                    subjName = ''
                else:
                    subjName = self.subjectName + '_'
                    self.saveDir = os.path.join(self.saveDir,self.subjectName)
                    if not os.path.exists(self.saveDir):
                        os.makedirs(self.saveDir)
                filePath = os.path.join(self.saveDir,self.__class__.__name__ + '_' + subjName + self.startTime)
                fileOut = h5py.File(filePath+'.hdf5','w')
                saveParameters(fileOut,self.__dict__)
                if self.saveFrameIntervals and self._win is not None:
                    fileOut.create_dataset('frameIntervals',data=self._win.frameIntervals)
                fileOut.close()
        
    
    def startNidaqDevice(self):
        # rotary encoder
        aiSampleRate = 2000 if self._win.monitorFramePeriod < 0.0125 else 1000
        aiBufferSize = 16
        self._rotaryEncoderInput = nidaqmx.Task()
        #self._rotaryEncoderInput.ai_channels.add_ai_voltage_chan(self.nidaqDeviceNames[0]+'/ai'+str(self.rotaryEncoderCh),min_val=0,max_val=5)
        self._rotaryEncoderInput.ai_channels.add_ai_voltage_chan(self.nidaqDeviceNames[0]+'/ai2',min_val=0,max_val=1)
        self._rotaryEncoderInput.timing.cfg_samp_clk_timing(aiSampleRate,
                                                            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                                            samps_per_chan=aiBufferSize)
                                            
        def readRotaryEncoderBuffer(task_handle,every_n_samples_event_type,number_of_samples,callback_data):
            self._rotaryEncoderData = self._rotaryEncoderInput.read(number_of_samples_per_channel=number_of_samples)
            return 0
        
        self._rotaryEncoderInput.register_every_n_samples_acquired_into_buffer_event(aiBufferSize,readRotaryEncoderBuffer)
        self._rotaryEncoderData = None
        self._rotaryEncoderInput.start()
        self._nidaqTasks.append(self._rotaryEncoderInput)
        
        # water reward solenoid
        self._rewardOutput = nidaqmx.Task()
        if self.digitalSolenoidTrigger:
            self._rewardOutput.do_channels.add_do_chan(self.nidaqDeviceNames[0]+'/port0/line7',
                                                       line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self._rewardOutput.write(False)
        else:
            aoSampleRate = 1000
            self._rewardOutput.ao_channels.add_ao_voltage_chan(self.nidaqDeviceNames[0]+'/ao0',min_val=0,max_val=5)
            self._rewardOutput.write(0)
            self._rewardOutput.timing.cfg_samp_clk_timing(aoSampleRate)
        self._nidaqTasks.append(self._rewardOutput)
            
        # lick input
        self._lickInput = nidaqmx.Task()
        self._lickInput.di_channels.add_di_chan(self.nidaqDeviceNames[0]+'/port0/line0',
                                                line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._nidaqTasks.append(self._lickInput)
        
        # frame signal
        self._frameSignalOutput = nidaqmx.Task()
        self._frameSignalOutput.do_channels.add_do_chan(self.nidaqDeviceNames[0]+'/port1/line0',
                                                        line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._frameSignalOutput.write(False)
        self._nidaqTasks.append(self._frameSignalOutput)
        
        if self.soundMode == 'external':
            # sound1 trigger
            self._sound1Output = nidaqmx.Task()
            self._sound1Output.do_channels.add_do_chan(self.nidaqDeviceNames[0]+'/port1/line1',
                                                       line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self._sound1Output.write(False)
            self._nidaqTasks.append(self._sound1Output)
            
            # sound2 trigger
            self._sound2Output = nidaqmx.Task()
            self._sound2Output.do_channels.add_do_chan(self.nidaqDeviceNames[0]+'/port1/line2',
                                                       line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self._sound2Output.write(False)
            self._nidaqTasks.append(self._sound2Output)
        
        # LEDs
        if len(self.nidaqDevices)>1 and self.nidaqDevices[1]=='USB-6001':
            self._optoOutput = nidaqmx.Task()
            self._optoOutput.ao_channels.add_ao_voltage_chan(self.nidaqDeviceNames[1]+'/ao0:1',min_val=0,max_val=5)
            self._optoOutput.write([0,0])
            self._optoAmp = 0
            self._optoOutput.timing.cfg_samp_clk_timing(aoSampleRate)
            self._nidaqTasks.append(self._optoOutput)
    
    
    def stopNidaqDevice(self):
        if getattr(self,'_optoAmp',0):
            self.optoOff()
        for task in self._nidaqTasks:
            task.close()
            
                
    def getNidaqData(self):
        # analog
        if self._rotaryEncoderData is None:
            self.rotaryEncoderVolts.append(np.nan)
            encoderAngle = np.nan
        else:
            #self.rotaryEncoderVolts.append(self._rotaryEncoderData[-1])
            self.rotaryEncoderVolts.append(np.std(self._rotaryEncoderData))
            encoderData = np.array(self._rotaryEncoderData)
            encoderData *= 2 * math.pi / 5
            encoderAngle = np.arctan2(np.mean(np.sin(encoderData)),np.mean(np.cos(encoderData)))
        self.wheelPosRadians.append(encoderAngle)
        self.deltaWheelPos.append(self.calculateWheelChange())
        
        # digital
        if self._lickInput.read():
            self._lick = True
            self.lickFrames.append(self._sessionFrame)
        else:
            self._lick = False    
        
    
    def calculateWheelChange(self):
        # calculate angular change in wheel position
        if len(self.wheelPosRadians) < 2 or np.isnan(self.wheelPosRadians[-1]):
            angleChange = 0
        else:
            angleChange = self.wheelPosRadians[-1] - self.wheelPosRadians[-2]
            if angleChange < -math.pi:
                angleChange += 2 * math.pi
            elif angleChange > math.pi:
                angleChange -= 2 * math.pi
            if self.minWheelAngleChange < abs(angleChange) < self.maxWheelAngleChange:
                angleChange *= self.wheelPolarity
            else:
                angleChange = 0
        return angleChange


    def initSolenoid(self):
        self._solenoid = nidaqmx.Task()
        if self.digitalSolenoidTrigger:
            self._solenoid.do_channels.add_do_chan(self.nidaqDeviceNames[0]+'/port0/line7',
                                                   line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        else:
            self._solenoid.ao_channels.add_ao_voltage_chan(self.nidaqDeviceNames[0]+'/ao0',min_val=0,max_val=5)
            
            
    def openSolenoid(self):
        self.initSolenoid()
        if self.digitalSolenoidTrigger:
            self._solenoid.write(True)
        else:
            self._solenoid.write(5)
        
    
    def closeSolenoid(self):
        if not getattr(self,'_solenoid',0):
            self.initSolenoid()
        if self.digitalSolenoidTrigger:
            self._solenoid.write(False)
        else:
            self._solenoid.write(0)
        self._solenoid.stop()
        self._solenoid.close()
        self._solenoid = None
        
        
    def triggerReward(self,openTime):
        if self.digitalSolenoidTrigger:
            t = Timer(openTime,self.endReward)
            self._rewardOutput.write(True)
            t.start()
        else:
            sampleRate = self._rewardOutput.timing.samp_clk_rate
            nSamples = int(openTime * sampleRate) + 1
            s = np.zeros(nSamples)
            s[:-1] = 5
            self._rewardOutput.stop()
            self._rewardOutput.timing.samp_quant_samp_per_chan = nSamples
            self._rewardOutput.write(s,auto_start=True)
    
    
    def endReward(self):
        if self.digitalSolenoidTrigger:
            self._rewardOutput.write(False)

          
    def optoOn(self,ch=[0,1],amp=5,ramp=0):
        self.optoPulse(ch,amp,onRamp=ramp,lastVal=amp)
    
    
    def optoOff(self,ch=[0,1],ramp=0):
        amp = self._optoAmp if ramp > 0 else 0 
        self.optoPulse(ch,amp,offRamp=ramp)
    
    
    def optoPulse(self,ch=[0,1],amp=5,dur=0,onRamp=0,offRamp=0,lastVal=0):
        sampleRate = self._optoOutput.timing.samp_clk_rate
        nSamples = int((dur + onRamp + offRamp) * sampleRate) + 1
        if nSamples < 2:
            nSamples = 2
        pulse = np.zeros((2,nSamples))
        pulse[ch,:-1] = amp
        pulse[ch,-1] = lastVal
        if onRamp > 0:
            ramp = np.linspace(0,amp,int(onRamp * sampleRate))
            pulse[ch,:ramp.size] = ramp
        if offRamp > 0:
            ramp = np.linspace(amp,0,int(offRamp * sampleRate))
            pulse[ch,-(ramp.size+1):-1] = ramp
        self._optoOutput.stop()
        self._optoOutput.timing.samp_quant_samp_per_chan = nSamples
        self._optoOutput.write(pulse,auto_start=True)
        self._optoAmp = lastVal
    
    
    def makeSoundArray(self,soundType,soundDur,soundVolume=1,toneFreq=None,hanningDur=0.005):
        if soundType == 'tone':
            soundArray = np.sin(2 * np.pi * toneFreq * np.arange(0,soundDur,1/self.soundSampleRate))
        elif soundType == 'noise':
            soundArray = 2 * np.random.random(soundDur*self.soundSampleRate) - 1
        soundArray *= soundVolume
        if hanningDur > 0: # reduce onset/offset click
            hanningSamples = int(self.soundSampleRate * hanningDur)
            hanningWindow = np.hanning(2 * hanningSamples + 1)
            soundArray[:hanningSamples] *= hanningWindow[:hanningSamples]
            soundArray[-hanningSamples:] *= hanningWindow[hanningSamples+1:]
        return soundArray

        
class WaterTest(TaskControl):
                
    def __init__(self,rigName,openTime=None,numPulses=100,pulseInterval=120):
        TaskControl.__init__(self,rigName)
        self.saveParams = False
        if openTime is not None:
            self.solenoidOpenTime = openTime
        self.numPulses = numPulses
        self.pulseInterval = pulseInterval
              
    def taskFlow(self):
        while self._continueSession:
            if self._sessionFrame > 0 and not self._sessionFrame % self.pulseInterval:
                if len(self.rewardFrames) < self.numPulses:
                    self._reward = self.solenoidOpenTime
                else:
                    self._continueSession = False
            self.showFrame()
            


class LuminanceTest(TaskControl):
                
    def __init__(self,rigName,levels=None,framesPerLevel=300):
        TaskControl.__init__(self,rigName)
        self.saveParams = False
        self.levels = np.arange(-1,1.1,0.25) if levels is None else levels
        self.framesPerLevel = framesPerLevel
              
    def taskFlow(self):
        i = 0
        while self._continueSession:
            if not self._sessionFrame % self.framesPerLevel:
                if i < len(self.levels):
                    self._win.color = self.levels[i]
                else:
                    self._continueSession = False
                i += 1
            self.showFrame()
        


def saveParameters(fileOut,paramDict,dictName=None):
    for key,val in paramDict.items():
        if key[0] != '_':
            if dictName is None:
                paramName = key
            else:
                paramName = dictName+'_'+key
            if isinstance(val,dict):
                saveParameters(fileOut,val,paramName)
            else:
                if val is None:
                    val = np.nan
                try:
                    if isinstance(val,(list,tuple)) and all(isinstance(v,str) for v in val):
                        fileOut.create_dataset(paramName,data=np.array(val,dtype=object),dtype=h5py.special_dtype(vlen=str))
                    else:
                        try:
                            fileOut.create_dataset(paramName,data=val)
                        except:
                            fileOut.create_dataset(paramName,data=np.array(val,dtype=object),dtype=h5py.special_dtype(vlen=float))
                except:
                    print('\n' + 'could not save ' + key)
                    

if __name__ == "__main__":
    import sys,json
    paramsPath = sys.argv[1]
    with open(paramsPath,'r') as f:
        params = json.load(f)
    if params['taskVersion'] == 'open solenoid':
        task = TaskControl(params['rigName'])
        task.openSolenoid()
    elif params['taskVersion'] == 'close solenoid':
        task = TaskControl(params['rigName'])
        task.closeSolenoid()
    elif params['taskVersion'] == 'water test':
        task = WaterTest(params['rigName'])
        task.start()
    elif params['taskVersion'] == 'luminance test':
        task = LuminanceTest(params['rigName'])
        task.start()
    elif params['taskVersion'] == 'sound test':
        #sampleRate = sounddevice.query_devices(sounddevice.default.device[1],'output')['default_samplerate']
        task = TaskControl(params['rigName'])
        soundType = 'noise'
        soundDur = 5
        soundVolume = 1
        toneFreq = 6000
        hanningDur = 0
        soundArray = task.makeSoundArray(soundType,soundDur,soundVolume,toneFreq,hanningDur)
        sounddevice.play(soundArray,task.soundSampleRate)
        sounddevice.wait()
    else:
        task = TaskControl(params['rigName'])
        task.saveParams = False
        task.maxFrames = 60 * 3600
        task.start()