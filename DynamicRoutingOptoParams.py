import numpy as np
import pandas as pd

def getBregmaGalvoData():
    f = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\OptoGui\NP3_bregma_galvo.txt"
    d = pd.read_csv(f,sep='\t')
    
    bregmaToGalvoFit = np.linalg.lstsq(np.concatenate((d[['bregma x','bregma y']],np.ones(d.shape[0])[:,None]),axis=1),
                                       d[['galvo x','galvo y']])[0].T

    galvoToBregmaFit = np.linalg.lstsq(np.concatenate((d[['galvo x','galvo y']],np.ones(d.shape[0])[:,None]),axis=1),
                                       d[['bregma x','bregma y']])[0].T
    
    return bregmaToGalvoFit,galvoToBregmaFit

bregmaToGalvoFit,galvoToBregmaFit = getBregmaGalvoData()

def bregmaToGalvo(bregmaX,bregmaY):
    galvoX = bregmaToGalvoFit[0,0]*bregmaX + bregmaToGalvoFit[0,1]*bregmaY + bregmaToGalvoFit[0,2]
    galvoY = bregmaToGalvoFit[1,0]*bregmaX + bregmaToGalvoFit[1,1]*bregmaY + bregmaToGalvoFit[1,2]
    return galvoX,galvoY

def galvoToBregma(galvoX,galvoY):
    bregmaX = galvoToBregmaFit[0,0]*galvoX + galvoToBregmaFit[0,1]*galvoY + galvoToBregmaFit[0,2]
    bregmaY = galvoToBregmaFit[1,0]*galvoX + galvoToBregmaFit[1,1]*galvoY + galvoToBregmaFit[1,2]
    return bregmaX,bregmaY


optoParams = {
              'test': {
                       'V1': {'optoVoltage': 2, 'bregma': (-3,-3)},
                       'PFC': {'optoVoltage': 2, 'bregma': (-1.5,2.5)},
                       'ACC': {'optoVoltage': 2, 'bregma': (-0.5,1)},
			                },
    
              '636761': {
                         'V1': {'optoVoltage': 5, 'galvoVoltage': (-0.2,-2.05)},
      			             'A1': {'optoVoltage': 5, 'galvoVoltage': (0.18,-1.75)},
      			             'PFC': {'optoVoltage': 5, 'galvoVoltage': (-0.2,-1.22)},
      			             'ACC': {'optoVoltage': 5, 'galvoVoltage': (-0.55,-1.4)},
			                  },

			        '636766': {
      			             'V1': {'optoVoltage': 5, 'galvoVoltage': (-0.17,-2.18)},
      			             'A1': {'optoVoltage': 2, 'galvoVoltage': (0.2,-1.85)},
      			             'PFC': {'optoVoltage': 2, 'galvoVoltage': (-0.2,-1.16)},
      			             'ACC': {'optoVoltage': 2, 'galvoVoltage': (-0.5,-1.35)},
			                  },

              '643280': {
                         'V1': {'optoVoltage': 5, 'bregma': (-3.5,-4.1)},
                         'ACC': {'optoVoltage': 5, 'bregma': (-0.75,1)},
                         'mFC': {'optoVoltage': 5, 'bregma': (-0.8,2.5)},
                         'lFC': {'optoVoltage': 5, 'bregma': (-2,2.5)},
                        },
			       }
