import os
import numpy as np
import scipy.signal
from scipy.interpolate import interpn, LinearNDInterpolator


# opto utils

optoBaseDir = r'\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\OptoGui'


def _txtToDict(f):
    with open(f,'r') as r:
        cols = zip(*[line.strip('\n').split('\t') for line in r.readlines()]) 
    return {d[0]: [float(s) for s in d[1:]] for d in cols}


def getBregmaGalvoCalibrationData(rigName):
    bregmaGalvoFile = os.path.join(optoBaseDir,rigName,rigName + '_bregma_galvo.txt')
    d = _txtToDict(bregmaGalvoFile)
    return d


def _bregmaToGalvoInterpolation(px,py,vx,vy,bregmaX,bregmaY):
    return [interpn((px,py),v,(bregmaX,bregmaY),bounds_error=False,fill_value=None)[0] for v in (vx,vy)]
  
  
def bregmaToGalvo(calibrationData,bregmaX,bregmaY,offsetX=0,offsetY=0):
    px = np.unique(calibrationData['bregmaX'])
    py = np.unique(calibrationData['bregmaY'])
    vx = np.zeros((len(px),len(py)))
    vy = vx.copy()
    for x,y,zx,zy in zip(calibrationData['bregmaX'],calibrationData['bregmaY'],calibrationData['galvoX'],calibrationData['galvoY']):
      i = np.where(px==x)[0][0]
      j = np.where(py==y)[0][0]
      vx[i,j] = zx
      vy[i,j] = zy
    galvoX,galvoY = _bregmaToGalvoInterpolation(px,py,vx,vy,bregmaX,bregmaY)
    if offsetX != 0 or offsetY != 0:
        x0,y0 = _bregmaToGalvoInterpolation(px,py,vx,vy,0,0)
        xOff,yOff = _bregmaToGalvoInterpolation(px,py,vx,vy,offsetX,offsetY)
        galvoX += xOff - x0
        galvoY += yOff - y0
    return galvoX,galvoY


def galvoToBregma(calibrationData,galvoX,galvoY):
    points = np.stack((calibrationData['galvoX'],calibrationData['galvoY']),axis=1)
    bregmaX,bregmaY = [float(LinearNDInterpolator(points,calibrationData[b])(galvoX,galvoY)) for b in ('bregmaX','bregmaY')]
    return bregmaX,bregmaY


def getOptoPowerCalibrationData(rigName,devName):
    f = os.path.join(optoBaseDir,rigName,rigName + '_' + devName + '_power.txt')
    d = _txtToDict(f)
    p = np.polyfit(d['input (V)'],d['power (mW)'],2)
    d['poly coefficients'] = p
    d['offsetV'] = min(np.roots(p))
    return d


def powerToVolts(calibrationData,power):
    return min((np.poly1d(calibrationData['poly coefficients']) - power).roots) if power > 0 else 0


def voltsToPower(calibrationData,volts):
    return np.polyval(calibrationData['poly coefficients'],volts)


def getOptoPulseWaveform(sampleRate,amp,dur=0,delay=0,freq=0,onRamp=0,offRamp=0,offset=0,lastVal=0):
    nSamples = int((dur + onRamp + offRamp) * sampleRate) + 1
    if nSamples < 2:
        nSamples = 2
    if freq > 0:
        t = np.arange(nSamples) / sampleRate
        waveform = np.sin(2 * np.pi * freq * t)
        waveform *= 0.5 * (amp - offset)
        waveform += 0.5 * (amp + offset)
    else:
        waveform = np.zeros(nSamples)
        waveform[:-1] = amp
    waveform[-1] = lastVal
    if onRamp > 0:
        ramp = np.linspace(offset,1,int(onRamp*sampleRate))
        waveform[:ramp.size] *= ramp
    if offRamp > 0:
        ramp = np.linspace(1,offset,int(offRamp*sampleRate))
        waveform[-(ramp.size+1):-1] *= ramp
    if delay > 0:
        waveform = np.concatenate((np.zeros(int(delay*sampleRate)),waveform))
    return waveform


def getGalvoWaveforms(sampleRate,x,y,dwellTime,nSamples):
    # x and y are lists of positions
    # dwell time is time spent at each position before repeating the cycle
    dwellSamples = int(dwellTime * sampleRate)
    nRepeats = int(np.ceil(nSamples / dwellSamples))
    galvoX,galvoY = np.tile(np.repeat(np.stack((x,y)),dwellSamples,axis=1),nRepeats)[:,:nSamples]
    return galvoX,galvoY


# sound utils

def makeSoundArray(soundType,sampleRate,dur,hanningDur,vol,freq,AM=None,seed=None):
    t = np.arange(0,dur,1/sampleRate)
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
        b,a = scipy.signal.butter(10,freq,btype='bandpass',fs=sampleRate)
        soundArray = scipy.signal.filtfilt(b,a,soundArray)
        soundArray = np.ascontiguousarray(soundArray)
    if AM is not None and ~np.isnan(AM) and AM > 0:
        soundArray *= (np.sin(1.5*np.pi + 2*np.pi*AM*t) + 1) / 2
    elif hanningDur > 0:
        # reduce onset/offset click
        hanningSamples = int(sampleRate * hanningDur)
        hanningWindow = np.hanning(2 * hanningSamples + 1)
        soundArray[:hanningSamples] *= hanningWindow[:hanningSamples]
        soundArray[-hanningSamples:] *= hanningWindow[hanningSamples+1:]
    soundArray /= np.absolute(soundArray).max()
    soundArray *= vol
    return soundArray


def dBToVol(dB,a,b,c):
    return np.log(1 - ((dB - c) / a)) / b


def volTodB(vol,a,b,c):
    return a * (1 - np.exp(vol * b)) + c