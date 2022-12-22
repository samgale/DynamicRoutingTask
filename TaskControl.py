# -*- coding: utf-8 -*-
"""
Superclass for behavioral task control

"""

from __future__ import division
import math, os, time
from threading import Timer
import h5py
import numpy as np
import scipy.signal
from psychopy import monitors, visual, event
from psychopy.visual.windowwarp import Warper
import psychtoolbox.audio
import sounddevice
import nidaqmx
import serial


class TaskControl():
    
    def __init__(self,params=None):
        self.configPath = None
        self.rigName = None
        self.subjectName = None
        self.startTime = None
        self.maxFrames = None # max number of frames before task terminates
        self.saveParams = True # if True, saves all attributes not starting with underscore
        self.saveFrameIntervals = True
        self.monBackgroundColor = 0 # gray; can adjust this for luminance measurement
        self.minWheelAngleChange = 0 # radians per frame
        self.maxWheelAngleChange = 0.5 # radians per frame
        self.spacebarRewardsEnabled = True
        self.soundMode = 'internal' # internal (sound card)
        self.soundLibrary = 'psychtoolbox' # 'psychtoolbox' or 'sounddevice'
        self.soundSampleRate = 48000 # Hz
        self.soundHanningDur = 0.005 # seconds
        
        # rig specific settings
        self.frameRate = 60
        self.screen = 0 # monitor to present stimuli on
        self.monWidth = 52.0 # cm
        self.monDistance = 15.3 # cm
        self.monGamma = 2.3 # float or None
        self.monSizePix = (1920,1200)
        self.warp = None # 'spherical', 'cylindrical', 'warpfile', None
        self.warpFile = None
        self.drawDiodeBox = False
        self.wheelRadius = 4.69 # cm
        self.wheelPolarity = -1
        self.rotaryEncoder = 'digital' # 'digital', 'analog', or None
        self.rotaryEncoderCh = 1 # nidaq analog input channel
        self.rotaryEncoderSerialPort = 'COM3' # serial input from arduino for digital encoder
        self.rotaryEncoderCountsPerRev = 8192 # digital pulses per revolution of encoder
        self.microphoneCh = None
        self.digitalSolenoidTrigger = True
        self.behavNidaqDevice = 'Dev1'
        self.syncNidaqDevice = None
        self.optoNidaqDevice = None
        self.galvoNidaqDevice = None
        self.soundNidaqDevice = None
        self.solenoidOpenTime = 0.03
        
        if params is not None:
            self.rigName = params['rigName']
            if 'configPath' in params:
                self.startTime = params['startTime']
                self.saveDir = None
                self.savePath = params['savePath']
                self.configPath = params['configPath']
                self.rotaryEncoderSerialPort = params['rotaryEncoderSerialPort']
                self.behavNidaqDevice = params['behavNidaqDevice']
                self.rewardVol = 0.003 # uL
                self.solenoidOpenTime = params['waterCalibrationSlope'] * self.rewardVol + params['waterCalibrationIntercept']
            else:
                self.saveDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data"
                if self.rigName == 'NP3':
                    self.drawDiodeBox = True
                    self.diodeBoxSize = 120
                    self.diodeBoxPosition = (900,540)
                    self.rotaryEncoderSerialPort = 'COM5'
                    self.behavNidaqDevice = 'Dev0'
                    self.syncNidaqDevice = 'Dev1'
                    self.optoNidaqDevice = 'Dev2'
                    self.galvoNidaqDevice = 'GalvoDAQ'
                elif self.rigName in ('B1','B2','B3','B4','B5','B6'):
                    if self.rigName == 'B1':
                        self.solenoidOpenTime = 0.02 # 3.0 uL
                    elif self.rigName == 'B2':
                        self.solenoidOpenTime = 0.03 # 2.2 uL
                    elif self.rigName == 'B3':
                        self.solenoidOpenTime = 0.03 # 2.7 uL
                    elif self.rigName == 'B4':
                        self.solenoidOpenTime = 0.02 # 3.3 uL
                    elif self.rigName == 'B5':
                        self.solenoidOpenTime = 0.015 # 2.9 uL
                    elif self.rigName == 'B6':
                        self.solenoidOpenTime = 0.03 # 2.3 uL
                else:
                    raise ValueError(self.rigName + ' is not a recognized rig name')
                
            
    def prepareSession(self,window=True):
        self._win = None
        self._nidaqTasks = []
        
        if self.startTime is None:
            startTime = time.localtime()
            self.startTime = time.strftime('%Y%m%d_%H%M%S',startTime)
            print('start time was: ' + time.strftime('%I:%M',startTime))
        
        self.pixelsPerDeg = 0.5 * self.monSizePix[0] / math.degrees(math.atan(0.5 * self.monWidth / self.monDistance))
        
        if window:
            self.prepareWindow()
        
        self.initSound()
        
        self.startNidaqDevice()

        if self.rotaryEncoder == 'digital':
            self.initDigitalEncoder()
        
        self.rotaryEncoderVolts = [] # rotary encoder analog input each frame
        self.rotaryEncoderIndex = [] # rotary encoder digital input read index
        self.rotaryEncoderCount = [] # rotary encoder digital input count
        self.wheelPosRadians = []
        self.deltaWheelPos = []
        self.microphoneData = []
        self.lickFrames = []
        
        self._continueSession = True
        self._sessionFrame = 0 # index of frame since start of session
        self._trialFrame = 0 # index of frame since start of trial
        self._reward = False # reward delivered at next frame flip if True
        self.rewardFrames = [] # index of frames at which reward delivered
        self.manualRewardFrames = [] # index of frames at which reward manually delivered
        self.rewardSize = [] # size (solenoid open time) of each reward
        self._sound = False # sound triggered at next frame flip if True
        self._galvo = False # False or galvo voltage waveform applied next frame flip
        self._opto = False # False or opto voltage waveform applied next frame flip
        
    
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
            self.getInputData()
            
            # do stuff, for example:
            # check for licks and/or wheel movement
            # update/draw stimuli
            
            self.showFrame()


    def getInputData(self):
        self.getNidaqData()
        if self.rotaryEncoder == 'digital':
            self.readDigitalEncoder()
    
    
    def showFrame(self):
        if self.syncNidaqDevice is not None:
            self._frameSignalOutput.write(True)
            if self._sessionFrame == 0:
                self._acquisitionSignalOutput.write(True)

        if self.spacebarRewardsEnabled and 'space' in event.getKeys(['space']) and not self._reward:
            self._reward = self.solenoidOpenTime
            self.manualRewardFrames.append(self._sessionFrame)
        
        escape = event.getKeys(['escape'],modifiers=True)
        if (len(escape) > 0 and escape[0][1]['shift']) or (self.maxFrames is not None and self._sessionFrame == self.maxFrames - 1):   
            self._continueSession = False

        # show new frame
        if self.drawDiodeBox:
            self._diodeBox.fillColor = -self._diodeBox.fillColor
            self._diodeBox.draw()
        self._win.flip()

        if self.syncNidaqDevice is not None:
            self._frameSignalOutput.write(False)
            if not self._continueSession:
                self._acquisitionSignalOutput.write(False)

        if self._sound:
            if self.soundMode == 'internal':
                self.playSound(self._sound[0])
            self._sound = False

        if self._galvo:
            self.applyGalvoWaveform(self._galvo[0])
            self._galvo = False
        
        if self._opto:
            self.applyOptoWaveform(self._opto[0])
            self._opto = False
        
        if self._reward:
            self.triggerReward(self._reward)
            self.rewardFrames.append(self._sessionFrame)
            self.rewardSize.append(self._reward)
            self._reward = False
           
        self._sessionFrame += 1
        self._trialFrame += 1
                                               
    
    def completeSession(self):
        try:
            if self._win is not None:
                self._win.close()
            self.stopNidaqDevice()
            if hasattr(self,'_audioStream'):
                self._audioStream.close()
            if hasattr(self,'_digitalEncoder'):
                self._digitalEncoder.close()
        except:
            raise
        finally:
            if self.saveParams:
                if self.subjectName is None:
                    subjName = ''
                else:
                    subjName = self.subjectName + '_'
                    if self.saveDir is not None:
                        saveDir = os.path.join(self.saveDir,self.subjectName)
                        if not os.path.exists(saveDir):
                            os.makedirs(saveDir)
                        self.savePath = os.path.join(saveDir,self.__class__.__name__ + '_' + subjName + self.startTime + '.hdf5')
                fileOut = h5py.File(self.savePath,'w')
                saveParameters(fileOut,self.__dict__)
                if self.saveFrameIntervals and self._win is not None:
                    fileOut.create_dataset('frameIntervals',data=self._win.frameIntervals)
                fileOut.close()
            self.startTime = None
                
    
    def initSound(self):
        if self.soundMode == 'internal':
            if self.soundLibrary == 'psychtoolbox':
                self._audioStream = psychtoolbox.audio.Stream(latency_class=[3],
                                                              freq=self.soundSampleRate,
                                                              channels=1)
            elif self.soundLibrary == 'sounddevice':
                sounddevice.default.latency = 0.016
                
    
    def playSound(self,soundArray):
        if self.soundLibrary == 'psychtoolbox':
            self._audioStream.fill_buffer(soundArray)
            self._audioStream.start()
        elif self.soundLibrary == 'sounddevice':
            sounddevice.play(soundArray,self.soundSampleRate)


    def stopSound(self):
        if self.soundLibrary == 'psychtoolbox':
            if hasattr(self,'_audioStream'):
                self._audioStream.stop()
        elif self.soundLibrary == 'sounddevice':
            sounddevice.stop()
                
                
    def makeSoundArray(self,soundType,dur,vol,freq,AM=None,seed=None):
        t = np.arange(0,dur,1/self.soundSampleRate)
        if soundType == 'tone':
            soundArray = np.sin(2 * np.pi * freq * t)
        elif soundType in ('linear sweep','log sweep'):
            f = np.linspace(freq[0],freq[1],t.size)
            if soundType == 'log sweep':
                f = (2 ** f) * 1000
            soundArray = np.sin(2 * np.pi * f * t)
        elif soundType in ('noise','AM noise'):
            rng = np.random.RandomState(seed)
            soundArray = 2 * rng.random(t.size) - 1
            b,a = scipy.signal.butter(10,freq,btype='bandpass',fs=self.soundSampleRate)
            soundArray = scipy.signal.filtfilt(b,a,soundArray)
        soundArray *= vol
        if AM is not None and ~np.isnan(AM) and AM > 0:
            soundArray *= (np.sin(1.5*np.pi + 2*np.pi*AM*t) + 1) / 2
        elif self.soundHanningDur > 0:
            # reduce onset/offset click
            hanningSamples = int(self.soundSampleRate * self.soundHanningDur)
            hanningWindow = np.hanning(2 * hanningSamples + 1)
            soundArray[:hanningSamples] *= hanningWindow[:hanningSamples]
            soundArray[-hanningSamples:] *= hanningWindow[hanningSamples+1:]
        return soundArray
        
    
    def startNidaqDevice(self):
        # rotary encoder and mircophone
        if self.rotaryEncoder == 'analog' or self.microphoneCh is not None:
            aiSampleRate = 2000 if self._win.monitorFramePeriod < 0.0125 else 1000
            aiBufferSize = 16
            self._analogInput = nidaqmx.Task()

            if self.rotaryEncoder == 'analog':
                self._analogInput.ai_channels.add_ai_voltage_chan(self.behavNidaqDevice+'/ai'+str(self.rotaryEncoderCh),
                                                                  min_val=0,max_val=5)
            if self.microphoneCh is not None:
                self._analogInput.ai_channels.add_ai_voltage_chan(self.behavNidaqDevice+'/ai'+str(self.microphoneCh),
                                                                  min_val=0,max_val=1)
            
            self._analogInput.timing.cfg_samp_clk_timing(aiSampleRate,
                                                         sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                                         samps_per_chan=aiBufferSize)
                                                
            def readAnalogInput(task_handle,every_n_samples_event_type,number_of_samples,callback_data):
                self._analogInputData = self._analogInput.read(number_of_samples_per_channel=number_of_samples)
                return 0
            
            self._analogInput.register_every_n_samples_acquired_into_buffer_event(aiBufferSize,readAnalogInput)
            self._analogInputData = None
            self._analogInput.start()
            self._nidaqTasks.append(self._analogInput)
        
        # water reward solenoid
        self._rewardOutput = nidaqmx.Task()
        if self.digitalSolenoidTrigger:
            self._rewardOutput.do_channels.add_do_chan(self.behavNidaqDevice+'/port0/line7',
                                                       line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self._rewardOutput.write(False)
        else:
            self._rewardOutput.ao_channels.add_ao_voltage_chan(self.behavNidaqDevice+'/ao0',min_val=0,max_val=5)
            self._rewardOutput.write(0)
            self._rewardOutput.timing.cfg_samp_clk_timing(1000) # samples/s
        self._nidaqTasks.append(self._rewardOutput)
            
        # lick input
        self._lickInput = nidaqmx.Task()
        self._lickInput.di_channels.add_di_chan(self.behavNidaqDevice+'/port0/line0',
                                                line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._nidaqTasks.append(self._lickInput)
        
        # frame signal
        if self.syncNidaqDevice is not None:
            self._frameSignalOutput = nidaqmx.Task()
            self._frameSignalOutput.do_channels.add_do_chan(self.syncNidaqDevice+'/port1/line4',
                                                            line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self._frameSignalOutput.write(False)
            self._nidaqTasks.append(self._frameSignalOutput)

            self._acquisitionSignalOutput = nidaqmx.Task()
            self._acquisitionSignalOutput.do_channels.add_do_chan(self.syncNidaqDevice+'/port1/line7',
                                                                  line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self._acquisitionSignalOutput.write(False)
            self._nidaqTasks.append(self._acquisitionSignalOutput)
        
        # LEDs/lasers
        if self.optoNidaqDevice is not None:
            self._optoOutput = nidaqmx.Task()
            self._optoOutput.ao_channels.add_ao_voltage_chan(self.optoNidaqDevice+'/ao0:1',min_val=0,max_val=5)
            self._optoVoltage = 0
            self._optoOutput.write([0,0])
            self._optoOutput.timing.cfg_samp_clk_timing(1000) # samples/s
            self._nidaqTasks.append(self._optoOutput)

        if self.galvoNidaqDevice is not None:
            self._galvoOutput = nidaqmx.Task()
            self._galvoOutput.ao_channels.add_ao_voltage_chan(self.galvoNidaqDevice+'/ao0:1',min_val=0,max_val=5)
            self._galvoVoltage = [1,1]
            self._galvoOutput.write(self._galvoVoltage)
            self._galvoOutput.timing.cfg_samp_clk_timing(1000) # samples/s
            self._nidaqTasks.append(self._galvoOutput)
    
    
    def stopNidaqDevice(self):
        if hasattr(self,'_optoVoltage'):
            self.optoOff()
        for task in self._nidaqTasks:
            task.close()
            
                
    def getNidaqData(self):
        # analog
        if hasattr(self,'_analogInput'):
            if self._analogInputData is None:
                if self.rotaryEncoder == 'analog':
                    self.rotaryEncoderVolts.append(np.nan)
                    encoderAngle = np.nan
                if self.microphoneCh is not None:
                    self.microphoneData.append(np.nan)
            else:
                if self.rotaryEncoder == 'analog':
                    encoderData = np.array(self._analogInputData) if self.microphoneCh is None else np.array(self._analogInputData[0])
                if self.microphoneCh is not None:
                    micData = self._analogInputData[1] if self.rotaryEncoder == 'analog' else self._analogInputData
                    self.microphoneData.append(np.std(micData))
                self.rotaryEncoderVolts.append(encoderData[-1])
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
        if not hasattr(self,'_solenoid'):
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

          
    def optoOn(self,amp,ramp=0):
        waveform = self.getOptoPulseWaveform(amp,onRamp=ramp,lastVal=amp)
        self.applyOptoWaveform(waveform)
    
    
    def optoOff(self,ramp=0):
        amp = self._optoVoltage if ramp > 0 else 0 
        waveform = self.getOptoPulseWaveform(amp,offRamp=ramp)
        self.applyOptoWaveform(waveform)
    
    
    def getOptoPulseWaveform(self,amp,dur=0,onRamp=0,offRamp=0,lastVal=0):
        sampleRate = self._optoOutput.timing.samp_clk_rate
        nSamples = int((dur + onRamp + offRamp) * sampleRate) + 1
        if nSamples < 2:
            nSamples = 2
        waveform = np.zeros(nSamples)
        waveform[:-1] = amp
        waveform[-1] = lastVal
        if onRamp > 0:
            ramp = np.linspace(0,amp,int(onRamp * sampleRate))
            waveform[:ramp.size] = ramp
        if offRamp > 0:
            ramp = np.linspace(amp,0,int(offRamp * sampleRate))
            waveform[-(ramp.size+1):-1] = ramp
        return waveform


    def applyOptoWaveform(self,waveform):
        self._optoOutput.stop()
        self._optoOutput.timing.samp_quant_samp_per_chan = waveform.size
        output = np.zeros((2,waveform.size))
        output[0] = waveform
        output[1,waveform>0] = 5
        self._optoOutput.write(output,auto_start=True)
        self._optoVoltage = waveform[-1]


    def getGalvoPositionWaveform(self,x=None,y=None):
        if x is None:
            x = self._galvoVoltage[0]
        if y is None:
            y = self._galvoVoltage[1]
        waveform = np.zeros((2,2))
        waveform[0] = x
        waveform[1] = y
        return waveform
    
    
    def setGalvoPosition(self,x,y):
        waveform = self.getGalvoPositionWaveform(x,y)
        self.applyGalvoWaveform(waveform)


    def applyGalvoWaveform(self,waveform):
        self._galvoOutput.stop()
        self._galvoOutput.timing.samp_quant_samp_per_chan = waveform.shape[1]
        self._galvoOutput.write(waveform,auto_start=True)
        self._galvoVoltage = waveform[:,-1]

    
    def initDigitalEncoder(self):
        self._digitalEncoder = serial.Serial(port=self.rotaryEncoderSerialPort,baudrate=9600,timeout=0.5)

        # intialize arduino
        for message,response in zip(('7','3','8'),('MDR0','STR','MDR0')):
            self._digitalEncoder.write(message.encode('utf8'))
            for _ in range(1000):
                val = self._digitalEncoder.readline()[:-2].decode('utf-8')
                if response in val:
                    break
            else:
                raise Exception('unable to initialize digital rotary encoder')

        # reset encoder count to zero
        self._digitalEncoder.write(b'2')
        count = 0
        val = self._digitalEncoder.readline()[:-2].decode('utf-8')
        c = int(val.split(';')[-1].split(':')[-1])
        while c > 1000 or c < -1000:
            count += 1
            if count == 1000:
                break
            val = self._digitalEncoder.readline()[:-2].decode('utf-8')
            c = int(val.split(';')[-1].split(':')[-1])

    
    def readDigitalEncoder(self):
        try:
            r = self._digitalEncoder.readline()[:-2].decode('utf-8')
            self.rotaryEncoderIndex.append(int(r.split(';')[-2].split(':')[-1]))
            self.rotaryEncoderCount.append(int(r.split(';')[-1].split(':')[-1]))
        except:
            self.rotaryEncoderIndex.append(np.nan)
            self.rotaryEncoderCount.append(np.nan)
    

        
class WaterTest(TaskControl):
                
    def __init__(self,params,openTime=None,numPulses=100,pulseInterval=120):
        TaskControl.__init__(self,params)
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
                
    def __init__(self,params,levels=None,framesPerLevel=300):
        TaskControl.__init__(self,params)
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
        


def saveParameters(group,paramDict):
    for key,val in paramDict.items():
        if key[0] != '_':
            if isinstance(val,dict):
                saveParameters(group.create_group(key),val)
            else:
                if val is None:
                    val = np.nan
                try:
                    if isStringSequence(val):
                        if not all(isinstance(v,str) for v in val):
                            for i,v in enumerate(val):
                                val[i] = str(v)
                        group.create_dataset(key,data=np.array(val,dtype=object),dtype=h5py.special_dtype(vlen=str)) 
                    elif (isinstance(val,(list,tuple,np.ndarray)) and len(val) > 0 and
                          all(isinstance(d,(list,tuple,np.ndarray)) for d in val) and [len(d) for d in val].count(len(val[0])) != len(val)):
                        group.create_dataset(key,data=np.array(val,dtype=object),dtype=h5py.special_dtype(vlen=float))
                    else:
                        group.create_dataset(key,data=val)
                except:
                    print('\n' + 'could not save ' + key)            


def isStringSequence(obj):
    if (isinstance(obj,(tuple,list,np.ndarray)) and len(obj) > 0 and
        all((isinstance(d,str) or isStringSequence(d)) for d in obj)):
        return True
    else:
        return False          


                    
if __name__ == "__main__":
    import sys,json
    paramsPath = sys.argv[1]
    with open(paramsPath,'r') as f:
        params = json.load(f)
    if params['taskVersion'] == 'open solenoid':
        task = TaskControl(params)
        task.openSolenoid()
    elif params['taskVersion'] == 'close solenoid':
        task = TaskControl(params)
        task.closeSolenoid()
    elif params['taskVersion'] == 'water test':
        task = WaterTest(params)
        task.start()
    elif params['taskVersion'] == 'luminance test':
        task = LuminanceTest(params)
        task.start()
    elif params['taskVersion'] == 'sound test':
        task = TaskControl(params)
        task.soundMode = 'internal'
        task.soundLibrary = 'psychtoolbox'
        task.initSound()
        soundDur = 4
        soundArray = task.makeSoundArray(soundType='tone',dur=soundDur,vol=0.1,freq=6000)
        task.playSound(soundArray)
        time.sleep(soundDur)
    else:
        task = TaskControl(params)
        task.saveParams = False
        task.maxFrames = 60 * 3600
        task.start()