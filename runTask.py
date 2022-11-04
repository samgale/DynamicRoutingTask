# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:48:22 2021

@author: svc_ccg
"""

import argparse
import json
import os
import subprocess

env = 'DynamicRoutingTaskDev'

parser = argparse.ArgumentParser()
parser.add_argument('params')
parser.add_argument('-o')

args = parser.parse_args()

paramsPath = args.params
paramsDir = os.path.dirname(paramsPath)

with open(paramsPath,'r') as f:
    params = json.load(f)
    
if 'rigName' not in params:
    import time
    import uuid
    import ConfigParser
    from camstim.zro.agent import CAMSTIM_CONFIG_PATH, OUTPUT_DIR
    from camstim.misc import CAMSTIM_CONFIG
    
    params['startTime'] = time.strftime('%Y%m%d_%H%M%S',time.localtime())
    params['savePath'] = os.path.join(OUTPUT_DIR,params['subjectName'] + '_' + params['startTime'] + '.hdf5')
    params['sessionID'] = uuid.uuid4().hex
    params['limsUpload'] = True
    params['configPath'] = CAMSTIM_CONFIG_PATH
    
    config = ConfigParser.ConfigParser()
    config.read(CAMSTIM_CONFIG_PATH)
    params['rotaryEncoderSerialPort'] = eval(config.get('DigitalEncoder','serial_device'))
    params['behavNidaqDevice'] = eval(config.get('Behavior','nidevice'))
    
    params['computerName'] = os.environ['aibs_comp_id']
    params['rigName'] = os.environ['aibs_rig_id']
    waterCalibration = CAMSTIM_CONFIG['shared']['water_calibration'][params['computerName']]
    params['waterCalibrationSlope'] = waterCalibration['slope']
    params['waterCalibrationIntercept'] = waterCalibration['intercept']
    
    paramsPath = os.path.join(paramsDir,'taskParams.json')
    with open(paramsPath,'w') as f:
        json.dump(params,f)
    
toRun = ('call activate ' + env + '\n' +
         'python ' + '"' + params['taskScript'] + '" ' + '"' + paramsPath + '"')

batFile = os.path.join(paramsDir,'toRun.bat')

with open(batFile,'w') as f:
    f.write(toRun)
    
p = subprocess.Popen([batFile])
p.wait()

# if 'limsUpload' in params and params['limsUpload']:
#     from camstim.lims import BehaviorSession
    
#     session = BehaviorSession(params['subjectName'],params['savePath'],params['sessionID'],params['userName'])
#     session.upload()
