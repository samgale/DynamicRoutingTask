import os
import numpy as np
import scipy.stats
from scipy.interpolate import interpn, LinearNDInterpolator


baseDir = r'\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\OptoGui'


def txtToDict(f):
    with open(f,'r') as r:
        cols = zip(*[line.strip('\n').split('\t') for line in r.readlines()]) 
    return {d[0]: [float(s) for s in d[1:]] for d in cols}


def getBregmaGalvoCalibrationData(rigName):
    f = os.path.join(baseDir,rigName + '_bregma_galvo.txt')
    return txtToDict(f)
  
  
def bregmaToGalvo(calibrationData,bregmaX,bregmaY):
    px = np.unique(calibrationData['bregmaX'])
    py = np.unique(calibrationData['bregmaY'])
    vx = np.zeros((len(px),len(py)))
    vy = vx.copy()
    for x,y,zx,zy in zip(calibrationData['bregmaX'],calibrationData['bregmaY'],calibrationData['galvoX'],calibrationData['galvoY']):
      i = np.where(px==x)[0][0]
      j = np.where(py==y)[0][0]
      vx[i,j] = zx
      vy[i,j] = zy
    galvoX,galvoY = [interpn((px,py),v,(bregmaX+bregmaOffset[0],bregmaY+bregmaOffset[1]),bounds_error=False,fill_value=None)[0] for v in (vx,vy)]
    return galvoX, galvoY


def galvoToBregma(calibrationData,galvoX,galvoY):
    points = np.stack((calibrationData['galvoX'],calibrationData['galvoY']),axis=1)
    bregmaX,bregmaY = [float(LinearNDInterpolator(points,calibrationData[b])(galvoX,galvoY)) for b in ('bregmaX','bregmaY')]
    return bregmaX+bregmaOffset[0], bregmaY+bregmaOffset[1]


def getOptoPowerCalibrationData(rigName,devName):
    f = os.path.join(baseDir,rigName + '_' + devName + '_power.txt')
    d = txtToDict(f)
    slope,intercept = scipy.stats.linregress(d['input (V)'],d['power (mW)'])[:2]
    d['slope'] = slope
    d['intercept'] = intercept
    return d


def powerToVolts(calibrationData,power):
    return (power - calibrationData['intercept']) / calibrationData['slope']


def voltsToPower(calibrationData,volts):
    return volts * calibrationData['slope'] + calibrationData['intercept']


bregmaOffset = (0.2,-0.2)


optoParams = {
              'test': {
                       'V1': {'power': 1.0, 'bregma': (-3.0,-3.0), 'use': True},
                       'PPC': {'power': 1.0, 'bregma': (-1.7,-2.0), 'use': True},
                       'pACC': {'power': 1.0, 'bregma': (-0.5,-0.5), 'use': True},
                       'ACC': {'power': 1.0, 'bregma': (-0.5,1.0), 'use': True},
                       'plFC': {'power': 1.0, 'bregma': (-2.0,1.0), 'use': True},
                       'mFC': {'power': 1.0, 'bregma': (-0.5,2.5), 'use': True},
                       'lFC': {'power': 1.0, 'bregma': (-2.0,2.5), 'use': True},
			                },

              '656726': {
                         'V1': {'power': 6.0, 'bregma': (-2.5,-3.7), 'use': True},
                         'V1_2': {'power': 6.0, 'bregma': (-2.75,-3.45), 'use': True},
                         'V1_3': {'power': 6.0, 'bregma': (-2.25,-4.2), 'use': True},
                         'PPC': {'power': 6.0, 'bregma': (-1.7,-2.0), 'use': False},
                         'pACC': {'power': 6.0, 'bregma': (-0.5,-0.5), 'use': False},
                         'ACC': {'power': 6.0, 'bregma': (-0.5,1.0), 'use': False},
                         'plFC': {'power': 6.0, 'bregma': (-2.0,1.0), 'use': False},
                         'mFC': {'power': 6.0, 'bregma': (-0.6,2.5), 'use': False},
                         'lFC': {'power': 6.0, 'bregma': (-2.0,2.5), 'use': False},
                        },

              }
