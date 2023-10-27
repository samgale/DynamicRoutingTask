# -*- coding: utf-8 -*-
"""
GUI for initiating camstim or samstim scripts

@author: SVC_CCG
"""

from __future__ import division
import os, sys, time
import subprocess
import numpy as np
from PyQt5 import QtCore, QtWidgets

sys.path.append(r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask")
from TaskUtils import getBregmaGalvoCalibrationData, galvoToBregma, bregmaToGalvo
from TaskUtils import getOptoPowerCalibrationData, powerToVolts, voltsToPower
from TaskUtils import getOptoPulseWaveform, getGalvoWaveforms
import TaskControl



def start():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    obj = OptoGui(app)
    app.exec_()


class OptoGui():
    
    def __init__(self,app):
        self.app = app
        self.baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"
        self.rigConfig = {'B2': {'computerName': ('wxvs-syslogic8',),
                                 'devNames': ('led_1','led_2','led_1,led_2'),
                                 'hasGalvos': False,
                                 'defaultGalvoXY': (0,0)},
                          'NP1': {'computerName': ('W10DT713843','W10DTSM18278','W10DT713942','w10dtsm18306'),
                                  'devNames': ('laser_488','laser_633','laser_488,laser_633'),
                                  'hasGalvos': True,
                                  'defaultGalvoXY': (2.4,0.1)},
                          'NP2': {'computerName': ('W10DT713844','W10DT714046','W10DT713937','w10dtsm18307'),
                                  'devNames': ('laser_488',),
                                  'hasGalvos': True,
                                  'defaultGalvoXY': (-1.1,0)},
                          'NP3': {'computerName': ('W10DT05516','W10DTSM118281','W10DT713941','W10DTSM118309'),
                                  'devNames': ('laser_488',),
                                  'hasGalvos': True,
                                  'defaultGalvoXY': (-0.2,2)}}
        self.rigNames = list(self.rigConfig.keys())
        self.defaultRig = 'NP3'
        for rig in self.rigNames:
            if os.environ['COMPUTERNAME'] in self.rigConfig[rig]['computerName']:
                self.defaultRig = rig
                break
        self.defaultGalvoXY = self.rigConfig[self.defaultRig]['defaultGalvoXY']
        self.defaultDwellTime = 0.005
        self.defaultAmpVolts = 0.4
        self.defaultFreq = 0
        self.defaultDur = 1
        self.useBregma = False
        self.usePower = False
        self.runAsTask = True
        self.task = None
        self.locTableColLabels = ('label','probability','device','bregmaX','bregmaY','dwell time','power','frequency','onset frame','delay','duration','on ramp','off ramp')
        self.locTableColLabelsOptotag = ('label','device','bregmaX','bregmaY')
        
        # control layout
        self.rigNameMenu = QtWidgets.QComboBox()
        self.rigNameMenu.addItems(self.rigNames)
        self.rigNameMenu.setCurrentIndex(self.rigNames.index(self.defaultRig))
        self.rigNameMenu.currentIndexChanged.connect(self.updateRig)
        
        self.devNameMenu = QtWidgets.QComboBox()
        self.devNameMenu.addItems(self.rigConfig[self.defaultRig]['devNames'])
        self.devNameMenu.currentIndexChanged.connect(self.updateDev)
        
        self.galvoButton = QtWidgets.QRadioButton('Galvo (V)')
        self.bregmaButton = QtWidgets.QRadioButton('Bregma (mm)')
        self.galvoButton.setChecked(True)
        self.galvoLayout = QtWidgets.QHBoxLayout()
        for button in (self.galvoButton,self.bregmaButton):
            button.clicked.connect(self.setGalvoMode)
            self.galvoLayout.addWidget(button)
        self.galvoGroupBox = QtWidgets.QGroupBox()
        self.galvoGroupBox.setLayout(self.galvoLayout)

        self.xLabel = QtWidgets.QLabel('X:')
        self.xLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.xEdit = QtWidgets.QLineEdit(str(self.defaultGalvoXY[0]))
        self.xEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.xEdit.editingFinished.connect(self.setXYValue)
        
        self.yLabel = QtWidgets.QLabel('Y:')
        self.yLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.yEdit = QtWidgets.QLineEdit(str(self.defaultGalvoXY[1]))
        self.yEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.yEdit.editingFinished.connect(self.setXYValue)
        
        self.dwellLabel = QtWidgets.QLabel('Dwell Time (ms):')
        self.dwellLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.dwellEdit = QtWidgets.QLineEdit(str(self.defaultDwellTime))
        self.dwellEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.dwellEdit.editingFinished.connect(self.setDwellValue)
        
        self.ampVoltsButton = QtWidgets.QRadioButton('Amplitude (V)')
        self.powerButton = QtWidgets.QRadioButton('Power (mW)')
        self.ampVoltsButton.setChecked(True)
        self.ampLayout = QtWidgets.QHBoxLayout()
        for button in (self.ampVoltsButton,self.powerButton):
            button.clicked.connect(self.setAmpMode)
            self.ampLayout.addWidget(button)
        self.ampGroupBox = QtWidgets.QGroupBox()
        self.ampGroupBox.setLayout(self.ampLayout)
        
        self.ampLabel = QtWidgets.QLabel('Amplitude (0-5 V):')
        self.ampLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.ampEdit = QtWidgets.QLineEdit(str(self.defaultAmpVolts))
        self.ampEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.ampEdit.editingFinished.connect(self.setAmpValue)

        self.freqLabel = QtWidgets.QLabel('Frequency (Hz):')
        self.freqLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.freqEdit = QtWidgets.QLineEdit(str(self.defaultFreq))
        self.freqEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.freqEdit.editingFinished.connect(self.setFreqValue)

        self.durLabel = QtWidgets.QLabel('Duration (s):')
        self.durLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.durEdit = QtWidgets.QLineEdit(str(self.defaultDur))
        self.durEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.durEdit.editingFinished.connect(self.setDurValue)

        self.runAsTaskButton = QtWidgets.QRadioButton('Run as task')
        self.directControlButton = QtWidgets.QRadioButton('Direct control')
        self.runAsTaskButton.setChecked(True)
        
        self.controlModeLayout = QtWidgets.QHBoxLayout()
        for button in (self.runAsTaskButton,self.directControlButton):
            button.clicked.connect(self.setControlMode)
            self.controlModeLayout.addWidget(button)
        self.controlModeGroupBox = QtWidgets.QGroupBox()
        self.controlModeGroupBox.setLayout(self.controlModeLayout)

        self.setOnOffButton = QtWidgets.QPushButton('Set On',checkable=True)
        self.setOnOffButton.setEnabled(False)
        self.setOnOffButton.clicked.connect(self.setOnOff)
        
        self.applyWaveformButton = QtWidgets.QPushButton('Apply Waveform')
        self.applyWaveformButton.clicked.connect(self.applyWaveform)
        
        self.controlLayout = QtWidgets.QGridLayout()
        self.controlLayout.addWidget(self.rigNameMenu,0,0,1,1)
        self.controlLayout.addWidget(self.devNameMenu,0,1,1,1)
        self.controlLayout.addWidget(self.galvoGroupBox,1,0,1,2)
        self.controlLayout.addWidget(self.xLabel,2,0,1,1)
        self.controlLayout.addWidget(self.xEdit,2,1,1,1)
        self.controlLayout.addWidget(self.yLabel,3,0,1,1)
        self.controlLayout.addWidget(self.yEdit,3,1,1,1)
        self.controlLayout.addWidget(self.dwellLabel,4,0,1,1)
        self.controlLayout.addWidget(self.dwellEdit,4,1,1,1)
        self.controlLayout.addWidget(self.ampGroupBox,5,0,1,2)
        self.controlLayout.addWidget(self.ampLabel,6,0,1,1)
        self.controlLayout.addWidget(self.ampEdit,6,1,1,1)
        self.controlLayout.addWidget(self.freqLabel,7,0,1,1)
        self.controlLayout.addWidget(self.freqEdit,7,1,1,1)
        self.controlLayout.addWidget(self.durLabel,8,0,1,1)
        self.controlLayout.addWidget(self.durEdit,8,1,1,1)
        self.controlLayout.addWidget(self.controlModeGroupBox,9,0,1,2)
        self.controlLayout.addWidget(self.setOnOffButton,10,0,1,1)
        self.controlLayout.addWidget(self.applyWaveformButton,10,1,1,1)
        
        # table layout
        self.mouseIdLabel = QtWidgets.QLabel('Mouse ID:')
        self.mouseIdLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.mouseIdEdit = QtWidgets.QLineEdit('')
        self.mouseIdEdit.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.locLabel = QtWidgets.QLabel('Label:')
        self.locLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.locEdit = QtWidgets.QLineEdit('')
        self.locEdit.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.optotagCheckbox = QtWidgets.QCheckBox('Optotagging')
        self.optotagCheckbox.setChecked(True)
        self.optotagCheckbox.clicked.connect(self.setOptotagging)
        
        self.addLocButton = QtWidgets.QPushButton('Add Location -->')
        self.addLocButton.setEnabled(False)
        self.addLocButton.clicked.connect(self.addLoc)

        self.useLocButton = QtWidgets.QPushButton('<-- Use Location')
        self.useLocButton.setEnabled(False)
        self.useLocButton.clicked.connect(self.useLoc)
        
        self.clearLocTableButton = QtWidgets.QPushButton('Clear Table')
        self.clearLocTableButton.clicked.connect(self.clearLocTable)
        
        self.loadLocTableButton = QtWidgets.QPushButton('Load Table')
        self.loadLocTableButton.clicked.connect(self.loadLocTable)
        
        self.locTable = QtWidgets.QTableWidget(0,len(self.locTableColLabelsOptotag))
        self.locTable.setHorizontalHeaderLabels(self.locTableColLabelsOptotag)

        self.calibrateXYCheckbox = QtWidgets.QCheckBox('Calibrate XY')
        self.calibrateXYCheckbox.clicked.connect(self.calibrateXY)

        self.testLocsButton = QtWidgets.QPushButton('Test Locations')
        self.testLocsButton.setEnabled(False)
        self.testLocsButton.clicked.connect(self.testLocs)
        
        self.saveLocTableButton = QtWidgets.QPushButton('Save Table')
        self.saveLocTableButton.clicked.connect(self.saveLocTable)
        
        self.locTableLayout = QtWidgets.QGridLayout()
        self.locTableLayout.addWidget(self.mouseIdLabel,0,0,1,2)
        self.locTableLayout.addWidget(self.mouseIdEdit,0,2,1,2)
        self.locTableLayout.addWidget(self.locLabel,0,4,1,2)
        self.locTableLayout.addWidget(self.locEdit,0,6,1,2)
        self.locTableLayout.addWidget(self.optotagCheckbox,0,8,1,4)
        self.locTableLayout.addWidget(self.addLocButton,1,0,1,6)
        self.locTableLayout.addWidget(self.useLocButton,2,0,1,6)
        self.locTableLayout.addWidget(self.clearLocTableButton,1,6,1,6)
        self.locTableLayout.addWidget(self.loadLocTableButton,2,6,1,6)
        self.locTableLayout.addWidget(self.locTable,3,0,6,12)
        self.locTableLayout.addWidget(self.calibrateXYCheckbox,9,1,1,3)
        self.locTableLayout.addWidget(self.testLocsButton,9,4,1,4)
        self.locTableLayout.addWidget(self.saveLocTableButton,9,8,1,4)
        
        # main window
        winHeight = 200
        winWidth = 610
        self.mainWin = QtWidgets.QMainWindow()
        self.mainWin.setWindowTitle('OptoGui')
        self.mainWin.closeEvent = self.mainWinClosed
        self.mainWin.resize(winWidth,winHeight)
        screenCenter = QtWidgets.QDesktopWidget().availableGeometry().center()
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(screenCenter)
        self.mainWin.move(mainWinRect.topLeft())
        
        self.mainWidget = QtWidgets.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayoutGridSpacing(self.mainLayout,winHeight,winWidth,1,3)
        self.mainLayout.addLayout(self.controlLayout,0,0,1,1)
        self.mainLayout.addLayout(self.locTableLayout,0,1,1,2)
        self.mainWidget.setLayout(self.mainLayout)
        
        self.mainWin.show()

        self.updateCalibrationData()

    def mainWinClosed(self,event):
        if self.task is not None:
            self.task.stopNidaqDevice()
        event.accept()

    def setLayoutGridSpacing(self,layout,height,width,rows,cols):
        for row in range(rows):
            layout.setRowMinimumHeight(row,int(height/rows))
            layout.setRowStretch(row,1)
        for col in range(cols):
            layout.setColumnMinimumWidth(col,int(width/cols))
            layout.setColumnStretch(col,1)
            
    def updateRig(self):
        if self.setOnOffButton.isChecked():
            self.setOff()
        self.devNameMenu.clear()
        self.devNameMenu.insertItems(0,self.rigConfig[self.rigNameMenu.currentText()]['devNames'])
        self.devNameMenu.setCurrentIndex(0)
    
    def updateDev(self):
        if self.setOnOffButton.isChecked():
            self.setOff()
        self.updateCalibrationData()

    def updateCalibrationData(self):
        rigName = self.rigNameMenu.currentText()
        self.hasGalvos = self.rigConfig[rigName]['hasGalvos']
        for item in (self.galvoButton,self.bregmaButton,self.xEdit,self.yEdit,self.dwellEdit):
            item.setEnabled(self.hasGalvos)
        if self.hasGalvos:
            try:
                self.bregmaGalvoCalibrationData = getBregmaGalvoCalibrationData(rigName)
                if not self.useBregma:
                    self.xEdit.setText(str(self.defaultGalvoXY[0]))
                    self.yEdit.setText(str(self.defaultGalvoXY[1]))
            except:
                self.bregmaGalvoCalibrationData = None
                self.bregmaButton.setEnabled(False)
                if self.useBregma:
                    self.useBregma = False
                    self.bregmaButton.setChecked(False)
                    self.galvoButton.setChecked(True)
                    self.changeGalvoMode()
        else:
            for item in (self.xEdit,self.yEdit):
                item.setText('nan')
            self.addLocButton.setEnabled(True)
        self.deviceNames = self.devNameMenu.currentText().split(',')
        self.calibrateXYCheckbox.setEnabled(len(self.deviceNames)==1)
        try:
            self.powerCalibrationData = {devName: getOptoPowerCalibrationData(rigName,devName) for devName in self.deviceNames}
            self.powerButton.setEnabled(True)
        except:
            self.powerCalibrationData = None
            self.powerButton.setEnabled(False)
            if self.usePower:
                self.usePower = False
                self.powerButton.setChecked(False)
                self.ampVoltsButton.setChecked(True)
                self.changeAmpMode()
    
    def setGalvoMode(self):
        sender = self.mainWin.sender()
        if (sender==self.galvoButton and self.useBregma) or (sender==self.bregmaButton and not self.useBregma):
            self.useBregma = not self.useBregma
            self.changeGalvoMode()
            
    def changeGalvoMode(self):
        if self.bregmaGalvoCalibrationData is None:
            self.xEdit.setText(str(self.defaultGalvoXY[0]))
            self.yEdit.setText(str(self.defaultGalvoXY[1]))
        else:
            xvals,yvals = [[float(val) for val in item.text().split(',')] for item in (self.xEdit,self.yEdit)]
            func = galvoToBregma if self.useBregma else bregmaToGalvo
            xvals,yvals = zip(*[func(self.bregmaGalvoCalibrationData,x,y) for x,y in zip(xvals,yvals)])
            self.xEdit.setText(','.join([str(round(x,3)) for x in xvals]))
            self.yEdit.setText(','.join([str(round(y,3)) for y in yvals]))
        for button in (self.addLocButton,self.useLocButton):
            button.setEnabled(self.useBregma)

    def setXYValue(self):
        sender = self.mainWin.sender()
        vals = [float(val) for val in sender.text().split(',')]
        if self.useBregma:
            minVal,maxVal = (-4,4) if sender is self.xEdit else (-5,4)
        else:
            minVal,maxVal = (-5,5)
        for i,val in enumerate(vals):
            if val < minVal:
                vals[i] = minVal
            elif val > maxVal:
                vals[i] = maxVal
        sender.setText(','.join([str(val) for val in vals]))
        if self.setOnOffButton.isChecked():
            self.setOn()
            
    def setDwellValue(self):
        val = float(self.dwellEdit.text())
        if val < 0:
            self.dwellEdit.setText('0')
                
    def setAmpMode(self):
        sender = self.mainWin.sender()
        if (sender==self.ampVoltsButton and self.usePower) or (sender==self.powerButton and not self.usePower):
            self.usePower = not self.usePower
            self.changeAmpMode()
            
    def changeAmpMode(self):
        label = 'Power (mW):' if self.usePower else 'Amplitude (0-5 V):'
        self.ampLabel.setText(label)
        if self.powerCalibrationData is None:
            self.ampEdit.setText(str(self.defaultAmpVolts))
        else:
            vals = [float(val) for val in self.ampEdit.text().split(',')]
            func = voltsToPower if self.usePower else powerToVolts
            vals = [func(self.powerCalibrationData[dev],val) for dev,val in zip(self.deviceNames,vals)]
            self.ampEdit.setText(','.join([str(round(val,3)) for val in vals]))
    
    def setAmpValue(self):
        vals = [float(val) for val in self.ampEdit.text().split(',')]
        if len(vals) > len(self.deviceNames):
            vals = vals[:len(self.deviceNames)]
        maxVolts = 5
        for i,(dev,val) in enumerate(zip(self.deviceNames,vals)):
            maxVal = voltsToPower(self.powerCalibrationData[dev],maxVolts) if self.usePower else maxVolts
            if val < 0:
                vals[i] = 0
            elif val > maxVal:
                vals[i] = maxVal
        self.ampEdit.setText(','.join([str(round(val,3)) for val in vals]))
        if self.setOnOffButton.isChecked():
            self.setOn()

    def setFreqValue(self):
        vals = [float(val) for val in self.freqEdit.text().split(',')]
        if len(vals) > len(self.deviceNames):
            vals = vals[:len(self.deviceNames)]
        for i,val in enumerate(vals):
            if val < 0:
                vals[i] = 0
        self.freqEdit.setText(','.join([str(val) for val in vals]))

    def setDurValue(self):
        vals = [float(val) for val in self.durEdit.text().split(',')]
        if len(vals) > len(self.deviceNames):
            vals = vals[:len(self.deviceNames)]
        for i,val in enumerate(vals):
            if val < 0:
                vals[i] = 0
        self.durEdit.setText(','.join([str(val) for val in vals]))

    def setControlMode(self):
        sender = self.mainWin.sender()
        if (sender==self.runAsTaskButton and not self.runAsTask) or (sender==self.directControlButton and self.runAsTask):
            self.runAsTask = not self.runAsTask
            if self.runAsTask:
                if self.setOnOffButton.isChecked():
                    self.task.optoOff()
                    self.setOnOffButton.setText('Set On')
                self.setOnOffButton.setEnabled(False)
                self.testLocsButton.setEnabled(False)
                if self.task is not None:
                    self.task.stopNidaqDevice()
                self.task = None              
            else:
                self.task = TaskControl.TaskControl(params={'rigName': self.rigNameMenu.currentText()})
                self.task.behavNidaqDevice = None
                self.task.syncNidaqDevice = None
                self.task._nidaqTasks = []
                self.task.startNidaqDevice()
                self.task.initOpto()
                self.setOnOffButton.setEnabled(True)
                self.testLocsButton.setEnabled(True)

    def setOnOff(self):
        if self.setOnOffButton.isChecked():
            self.setOn()
            self.setOnOffButton.setText('Set Off')
            self.applyWaveformButton.setEnabled(False)
            self.testLocsButton.setEnabled(False)
        else:
            self.setOff()
            self.setOnOffButton.setText('Set On')
            self.applyWaveformButton.setEnabled(True)
            self.testLocsButton.setEnabled(True)

    def setOn(self):
        amps = [float(val) for val in self.ampEdit.text().split(',')]
        if len(amps) == 1 and len(self.deviceNames) > 1:
            amps *= len(self.deviceNames)
        if self.usePower:
            amps = [powerToVolts(self.powerCalibrationData[dev],amp) for dev,amp in zip(self.deviceNames,amps)]
        if self.hasGalvos:
            x,y = self.getGalvoXY()
            x = x[0]
            y = y[0]
        else:
            x = y = None
        self.task.optoOn(self.deviceNames,amps,x=x,y=y)

    def setOff(self):
        self.task.optoOff(self.deviceNames)

    def applyWaveform(self):
        if self.runAsTask:
            self.startTask()
        else:
            optoWaveforms = self.getOptoWaveforms()
            nSamples = max(w.size for w in optoWaveforms)
            if self.hasGalvos:
                galvoVoltage = np.stack(self.getGalvoXY()).T
                dwellTime = float(self.dwellEdit.text())
                galvoX,galvoY = getGalvoWaveforms(self.task.optoSampleRate,galvoVoltage,dwellTime,nSamples)
            else:
                galvoX = galvoY = None
            dur = max([float(val) for val in self.durEdit.text().split(',')])
            self.task.loadOptoWaveform(self.deviceNames,optoWaveforms,galvoX,galvoY)
            self.task.startOpto()
            time.sleep(dur + 0.5)
            
    def getOptoParams(self):
        amps,freqs,durs = [[float(val) for val in item.text().split(',')] for item in (self.ampEdit,self.freqEdit,self.durEdit)]
        for vals in (amps,freqs,durs):
            if len(vals) == 1 and len(self.deviceNames) > 1:
                vals *= len(self.deviceNames)
            elif len(self.deviceNames) == 1 and len(vals) > 1:
                del vals[1:]
        if self.usePower:
            for i,(dev,amp,freq) in enumerate(zip(self.deviceNames,amps,freqs)):
                if freq > 0:
                    amp *= 2
                amps[i] = powerToVolts(self.powerCalibrationData[dev],amp)
            offsets = [self.powerCalibrationData[dev]['offsetV'] for dev in self.deviceNames]
        else:
            offsets = [0] * len(self.deviceNames)
        return amps,freqs,durs,offsets
    
    def getOptoWaveforms(self):
        return [getOptoPulseWaveform(self.task.optoSampleRate,amp,dur=dur,freq=freq,offset=offset) for amp,freq,dur,offset in zip(*self.getOptoParams())]
    
    def getGalvoXY(self):
        xvals,yvals = [[float(val) for val in item.text().split(',')] for item in (self.xEdit,self.yEdit)]
        if self.useBregma:
            xvals,yvals = zip(*[bregmaToGalvo(self.bregmaGalvoCalibrationData,x,y) for x,y in zip(xvals,yvals)])
        return xvals,yvals
    
    def startTask(self):
        rigName = self.rigNameMenu.currentText()
        scriptPath = os.path.join(self.baseDir,'startTask.py')
        taskScript = os.path.join(self.baseDir,'TaskControl.py')
        taskVersion = 'opto test'
        amp,freq,dur,offset = [','.join([str(val) for val in params]) for params in self.getOptoParams()]
        batString = ('python ' + '"' + scriptPath +'"' + 
                     ' --rigName ' + '"' + rigName + '"' + 
                     ' --taskScript ' + '"' + taskScript + '"' + 
                     ' --taskVersion ' + '"' + taskVersion + '"' +
                     ' --optoDev ' + ','.join(self.deviceNames) +
                     ' --optoAmp ' + amp +
                     ' --optoFreq ' + freq +
                     ' --optoDur ' + dur +
                     ' --optoOffset ' + offset)
        if self.hasGalvos:
            x,y = [','.join([str(val) for val in vals]) for vals in self.getGalvoXY()]
            dwell = self.dwellEdit.text()
            batString += (' --galvoX ' + x +
                          ' --galvoY ' + y +
                          ' --galvoDwellTime ' + dwell)
        self.runBatFile(batString)

    def runBatFile(self,batString):
        toRun = ('call activate zro27' + '\n' +
                 batString)

        batFile = os.path.join(self.baseDir,'samstimRun.bat')

        with open(batFile,'w') as f:
            f.write(toRun)
            
        p = subprocess.Popen([batFile])
        p.wait()
        
    def setOptotagging(self):
        self.locTable.setRowCount(0)
        colLabels = self.locTableColLabelsOptotag if self.optotagCheckbox.isChecked() else self.locTableColLabels
        self.locTable.setColumnCount(len(colLabels))
        self.locTable.setHorizontalHeaderLabels(colLabels)
        
    def addLoc(self):
        x = self.xEdit.text()
        y = self.yEdit.text()
        if self.calibrateXYCheckbox.isChecked():
            colLabels = [self.locTable.horizontalHeaderItem(col).text() for col in range(self.locTable.columnCount())]
            xcol = colLabels.index('galvoX')
            ycol = colLabels.index('galvoY')
            self.locTable.item(self.locTable.currentRow(),xcol).setText(x)
            self.locTable.item(self.locTable.currentRow(),ycol).setText(y)
        else:
            row = self.locTable.rowCount()
            lbl = self.locEdit.text()
            dev = self.devNameMenu.currentText()
            if self.optotagCheckbox.isChecked():
                vals = (lbl,dev,x,y)
            else:
                prob = 0.33 if row==0 else 'nan'
                dwell = self.defaultDwellTime
                power = 6
                freq = self.defaultFreq
                onset = 0
                delay = 0
                dur = 1
                onRamp = 0
                offRamp = 0.1
                vals = (lbl,prob,dev,x,y,dwell,power,freq,onset,delay,dur,onRamp,offRamp)
            self.locTable.insertRow(row)
            for col,val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)
                self.locTable.setItem(row,col,item)

    def useLoc(self):
        row = self.locTable.currentRow()
        colLabels = [self.locTable.horizontalHeaderItem(col).text() for col in range(self.locTable.columnCount())]
        if self.calibrateXYCheckbox.isChecked():
            xcol = colLabels.index('galvoX')
            ycol = colLabels.index('galvoY')
        else:
            xcol = colLabels.index('bregmaX')
            ycol = colLabels.index('bregmaY')
        self.xEdit.setText(self.locTable.item(row,xcol).text())
        self.yEdit.setText(self.locTable.item(row,ycol).text())
        if self.setOnOffButton.isChecked():
            self.setOn()

    def clearLocTable(self):
        self.locTable.setRowCount(0)

    def loadLocTable(self):
        dirName = 'optotagging' if self.optotagCheckbox.isChecked() else 'optoParams'
        filePath,fileType = QtWidgets.QFileDialog.getOpenFileName(self.mainWin,'Choose File',os.path.join(self.baseDir,'OptoGui',dirName),'*.txt',options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filePath == '':
            return
        if len(self.mouseIdEdit.text()) == 0:
            fileName = os.path.basename(filePath)
            i = fileName.find('_')
            mouseId = fileName[i+1:i+7]
            self.mouseIdEdit.setText(mouseId)
        self.locTable.setRowCount(0)
        with open(filePath,'r') as f:
            d = [line.strip('\n').split('\t') for line in f.readlines()][1:]
        for row in range(len(d)):
            self.locTable.insertRow(row)
            for col in range(self.locTable.columnCount()):
                item = QtWidgets.QTableWidgetItem(d[row][col])
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)
                self.locTable.setItem(row,col,item)
    
    def calibrateXY(self):
        self.locTable.setRowCount(0)
        if self.calibrateXYCheckbox.isChecked():
            self.galvoButton.setChecked(True)
            self.addLocButton.setText('Edit Location -->')
            self.addLocButton.setEnabled(True)
            self.useLocButton.setEnabled(True)
            self.clearLocTableButton.setEnabled(False)
            self.loadLocTableButton.setEnabled(False)
            colLabels = ('bregmaX','bregmaY','galvoX','galvoY')
            self.locTable.setColumnCount(len(colLabels))
            self.locTable.setHorizontalHeaderLabels(colLabels)
            for row in range(len(self.bregmaGalvoCalibrationData[colLabels[0]])):
                self.locTable.insertRow(row)
                for col,lbl in enumerate(colLabels):
                    item = QtWidgets.QTableWidgetItem(str(self.bregmaGalvoCalibrationData[lbl][row]))
                    item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)
                    self.locTable.setItem(row,col,item)
            self.xEdit.setText(str(self.bregmaGalvoCalibrationData['galvoX'][0]))
            self.yEdit.setText(str(self.bregmaGalvoCalibrationData['galvoY'][0]))
            self.useBregma = False
        else:
            self.addLocButton.setText('Add Location -->')
            self.addLocButton.setEnabled(False)
            self.useLocButton.setEnabled(False)
            self.clearLocTableButton.setEnabled(True)
            self.loadLocTableButton.setEnabled(True)
            colLabels = self.locTableColLabelsOptotag if self.optotagCheckbox.isChecked() else self.locTableColLabels
            self.locTable.setColumnCount(len(colLabels))
            self.locTable.setHorizontalHeaderLabels(colLabels)

    def testLocs(self):
        optoWaveforms = self.getOptoWaveforms()
        nSamples = max(w.size for w in optoWaveforms)
        dur = max([float(val) for val in self.durEdit.text().split(',')])
        colLabels = [self.locTable.horizontalHeaderItem(col).text() for col in range(self.locTable.columnCount())]
        xcol = colLabels.index('bregmaX')
        ycol = colLabels.index('bregmaY')
        for row in range(self.locTable.rowCount()):
            xvals,yvals = [[float(val) for val in self.locTable.item(row,col).text().split(',')] for col in (xcol,ycol)]
            xvals,yvals = zip(*[bregmaToGalvo(self.bregmaGalvoCalibrationData,x,y) for x,y in zip(xvals,yvals)])
            if self.optotagCheckbox.isChecked():                        
                galvoX,galvoY = [np.full(nSamples,vals[0]) for vals in (xvals,yvals)]
            else:
                galvoVoltage = np.stack((xvals,yvals)).T
                dwellTime = float(self.locTable.item(row,colLabels.index('dwell time')).text())
                galvoX,galvoY = getGalvoWaveforms(self.task.optoSampleRate,galvoVoltage,dwellTime,nSamples)
            self.task.loadOptoWaveform(self.deviceNames,optoWaveforms,galvoX,galvoY)
            self.task.startOpto()
            time.sleep(dur + 0.5)
            
    def saveLocTable(self):
        if self.calibrateXYCheckbox.isChecked():
            rigName = self.rigNameMenu.currentText()
            filePath = os.path.join(self.baseDir,'OptoGui',rigName,rigName + '_bregma_galvo.txt')
        else:
            baseName = 'optotagging' if self.optotagCheckbox.isChecked() else 'optoParams'
            fileName = (baseName + '_' +
                        self.mouseIdEdit.text() + '_' +
                        self.rigNameMenu.currentText() + '_' +
                        time.strftime('%Y%m%d_%H%M%S',time.localtime()) + '.txt')
            filePath = os.path.join(self.baseDir,'OptoGui',baseName,fileName)
        ncols = self.locTable.columnCount()
        colLabels = [self.locTable.horizontalHeaderItem(col).text() for col in range(ncols)]
        with open(filePath,'w') as f:
            for i,lbl in enumerate(colLabels):
                f.write(lbl)
                if i < ncols-1:
                    f.write('\t')
            for row in range(self.locTable.rowCount()):
                f.write('\n')
                for col in range(ncols):
                    f.write(self.locTable.item(row,col).text())
                    if col < ncols-1:
                        f.write('\t')
        if self.calibrateXYCheckbox.isChecked():
            self.bregmaGalvoCalibrationData = getBregmaGalvoCalibrationData(self.rigNameMenu.currentText())

                

if __name__=="__main__":
    start()