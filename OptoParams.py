import os
import numpy as np
import pandas as pd
from scipy.interpolate import interpn, LinearNDInterpolator


def getBregmaGalvoCalibrationData(rigName):
    f = os.path.join(r'\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\OptoGui',rigName+'_bregma_galvo.txt')
    d = pd.read_csv(f,sep='\t')
    return {col: np.array(d[col]) for col in ('bregmaX','bregmaY','galvoX','galvoY')}
    

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
    galvoX,galvoY = [interpn((px,py),v,(bregmaX,bregmaY))[0] for v in (vx,vy)]
    return galvoX, galvoY


def galvoToBregma(calibrationData,galvoX,galvoY):
    points = np.stack((calibrationData['galvoX'],calibrationData['galvoY']),axis=1)
    bregmaX,bregmaY = [float(LinearNDInterpolator(points,calibrationData[b])(galvoX,galvoY)) for b in ('bregmaX','bregmaY')]
    return bregmaX, bregmaY


optoParams = {
              'test': {
                       'V1': {'optoVoltage': 1, 'bregma': (-3,-3)},
                       'ACC': {'optoVoltage': 1, 'bregma': (-0.5,1)},
                       'mFC': {'optoVoltage': 1, 'bregma': (-0.5,2.5)},
                       'lFC': {'optoVoltage': 1, 'bregma': (-2,2.5)},
			                },

              '643280': {
                         'V1': {'optoVoltage': 5, 'bregma': (-3.5,-4.1)},
                         'ACC': {'optoVoltage': 5, 'bregma': (-0.75,1)},
                         'mFC': {'optoVoltage': 5, 'bregma': (-0.8,2.5)},
                         'lFC': {'optoVoltage': 5, 'bregma': (-2,2.5)},
                        },

			       }
