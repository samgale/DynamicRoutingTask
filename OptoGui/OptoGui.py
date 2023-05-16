# -*- coding: utf-8 -*-
"""
GUI for initiating camstim or samstim scripts

@author: SVC_CCG
"""

from __future__ import division
import os,sys
import subprocess
from PyQt5 import QtCore, QtWidgets

sys.path.append(r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask")
from OptoParams import getBregmaGalvoCalibrationData, galvoToBregma, bregmaToGalvo
from OptoParams import getOptoPowerCalibrationData, powerToVolts, voltsToPower



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
        
        winHeight = 200
        winWidth = 200
        self.mainWin = QtWidgets.QMainWindow()
        self.mainWin.setWindowTitle('OptoGui')
        self.mainWin.resize(winWidth,winHeight)
        screenCenter = QtWidgets.QDesktopWidget().availableGeometry().center()
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(screenCenter)
        self.mainWin.move(mainWinRect.topLeft())
        
        self.rigNameMenu = QtWidgets.QComboBox()
        self.rigNameMenu.addItems(('NP3',))
        self.rigNameMenu.currentIndexChanged.connect(self.updateRigDev)
        
        self.devNameMenu = QtWidgets.QComboBox()
        self.devNameMenu.addItems(('laser_488',))
        self.devNameMenu.currentIndexChanged.connect(self.updateRigDev)
        
        self.galvoButton = QtWidgets.QRadioButton('Galvo (V)')
        self.bregmaButton = QtWidgets.QRadioButton('Bregma (mm)')
        self.galvoButton.setChecked(True)
        self.useBregma = False
        self.galvoLayout = QtWidgets.QHBoxLayout()
        for button in (self.galvoButton,self.bregmaButton):
            button.clicked.connect(self.setGalvoMode)
            self.galvoLayout.addWidget(button)
        self.galvoGroupBox = QtWidgets.QGroupBox()
        self.galvoGroupBox.setLayout(self.galvoLayout)

        self.xLabel = QtWidgets.QLabel('X:')
        self.xLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.xEdit = QtWidgets.QLineEdit('-0.2')
        self.xEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.xEdit.editingFinished.connect(self.setXYValue)
        
        self.yLabel = QtWidgets.QLabel('Y:')
        self.yLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.yEdit = QtWidgets.QLineEdit('-2')
        self.yEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.yEdit.editingFinished.connect(self.setXYValue)
        
        self.inputVoltsButton = QtWidgets.QRadioButton('Input (V)')
        self.powerButton = QtWidgets.QRadioButton('Power (mW)')
        self.inputVoltsButton.setChecked(True)
        self.usePower = False
        self.ampLayout = QtWidgets.QHBoxLayout()
        for button in (self.inputVoltsButton,self.powerButton):
            button.clicked.connect(self.setAmpMode)
            self.ampLayout.addWidget(button)
        self.ampGroupBox = QtWidgets.QGroupBox()
        self.ampGroupBox.setLayout(self.ampLayout)
        
        self.ampLabel = QtWidgets.QLabel('Amplitude (0-5 V):')
        self.ampLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.ampEdit = QtWidgets.QLineEdit('0.4')
        self.ampEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.ampEdit.editingFinished.connect(self.setAmpValue)

        self.durLabel = QtWidgets.QLabel('Duration (s):')
        self.durLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.durEdit = QtWidgets.QLineEdit('1')
        self.durEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.durEdit.editingFinished.connect(self.setDurValue)
        
        self.applyValuesButton = QtWidgets.QPushButton('Apply Values')
        self.applyValuesButton.clicked.connect(self.startTask)

        self.mainWidget = QtWidgets.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayoutGridSpacing(self.mainLayout,winHeight,winWidth,8,2)
        self.mainLayout.addWidget(self.rigNameMenu,0,0,1,1)
        self.mainLayout.addWidget(self.devNameMenu,0,1,1,1)
        self.mainLayout.addWidget(self.galvoGroupBox,1,0,1,2)
        self.mainLayout.addWidget(self.xLabel,2,0,1,1)
        self.mainLayout.addWidget(self.xEdit,2,1,1,1)
        self.mainLayout.addWidget(self.yLabel,3,0,1,1)
        self.mainLayout.addWidget(self.yEdit,3,1,1,1)
        self.mainLayout.addWidget(self.ampGroupBox,4,0,1,2)
        self.mainLayout.addWidget(self.ampLabel,5,0,1,1)
        self.mainLayout.addWidget(self.ampEdit,5,1,1,1)
        self.mainLayout.addWidget(self.durLabel,6,0,1,1)
        self.mainLayout.addWidget(self.durEdit,6,1,1,1)
        self.mainLayout.addWidget(self.applyValuesButton,7,0,1,2)
        self.mainWidget.setLayout(self.mainLayout)
        
        self.mainWin.show()

        self.updateCalibrationData()

    def setLayoutGridSpacing(self,layout,height,width,rows,cols):
        for row in range(rows):
            layout.setRowMinimumHeight(row,int(height/rows))
            layout.setRowStretch(row,1)
        for col in range(cols):
            layout.setColumnMinimumWidth(col,int(width/cols))
            layout.setColumnStretch(col,1)

    def updateCalibrationData(self):
        self.bregmaGalvoCalibrationData = getBregmaGalvoCalibrationData(self.rigNameMenu.currentText())
        self.powerCalibrationData = getOptoPowerCalibrationData(self.rigNameMenu.currentText(),self.devNameMenu.currentText())
        
    def updateRigDev(self):
        sender = self.mainWin.sender()
        i = sender.currentIndex()
        if i > 0:
            self.sender.setCurrentIndex(0)
            self.updateCalibrationData()
    
    def setGalvoMode(self):
        sender = self.mainWin.sender()
        if (sender==self.galvoButton and self.useBregma) or (sender==self.bregmaButton and not self.useBregma):
            self.useBregma = not self.useBregma
            x = float(self.xEdit.text())
            y = float(self.yEdit.text())
            x,y = galvoToBregma(self.bregmaGalvoCalibrationData,x,y) if self.useBregma else bregmaToGalvo(self.bregmaGalvoCalibrationData,x,y)
            self.xEdit.setText(str(round(x,3)))
            self.yEdit.setText(str(round(y,3)))

    def setXYValue(self):
        sender = self.mainWin.sender()
        val = float(sender.text())
        if self.useBregma:
            d = self.bregmaGalvoCalibrationData['bregmaX'] if sender==self.xEdit else self.bregmaGalvoCalibrationData['bregmaY']
            low = d.min()
            high = d.max()
            if val < low:
                sender.setText(str(low))
            elif val > high:
                sender.setText(str(high))
        else:
            if val < -5:
                sender.setText('-5')
            elif val > 5:
                sender.setText('5')
                
    def setAmpMode(self):
        sender = self.mainWin.sender()
        if (sender==self.inputVoltsButton and self.usePower) or (sender==self.powerButton and not self.usePower):
            self.usePower = not self.usePower
            label = 'Power (mW):' if self.usePower else 'Amplitude (0-5 V):'
            self.ampLabel.setText(label)
            val = float(self.ampEdit.text())
            val = voltsToPower(self.powerCalibrationData,val) if self.usePower else powerToVolts(self.powerCalibrationData,val)
            self.ampEdit.setText(str(round(val,3)))
    
    def setAmpValue(self):
        val = float(self.ampEdit.text())
        high = voltsToPower(self.powerCalibrationData,5) if self.usePower else 5
        if val < 0:
            self.ampEdit.setText('0')
        elif val > high:
            self.ampEdit.setText(str(round(high,3)))

    def setDurValue(self):
        val = float(self.durEdit.text())
        if val < 0:
            self.durEdit.setText('0')

    def startTask(self):
        rigName = self.rigNameMenu.currentText()
        scriptPath = os.path.join(self.baseDir,'OptoGui','startOptoTask.py')
        taskScript = os.path.join(self.baseDir,'TaskControl.py')
        taskVersion = 'opto test'
        x = self.xEdit.text()
        y = self.yEdit.text()
        if self.useBregma:
            x,y = [str(n) for n in bregmaToGalvo(self.bregmaGalvoCalibrationData,float(x),float(y))]
        amp = self.ampEdit.text()
        if self.usePower:
            amp = str(powerToVolts(self.powerCalibrationData,float(amp)))
        dur = self.durEdit.text()
        batString = ('python ' + '"' + scriptPath +'"' + 
                     ' --rigName ' + '"' + rigName + '"' + 
                     ' --taskScript ' + '"' + taskScript + '"' + 
                     ' --taskVersion ' + '"' + taskVersion + '"' +
                     ' --galvoX ' + x +
                     ' --galvoY ' + y +
                     ' --optoAmp ' + amp +
                     ' --optoDur ' + dur)
        self.runBatFile(batString)

    def runBatFile(self,batString):
        toRun = ('call activate zro27' + '\n' +
                 batString)

        batFile = os.path.join(self.baseDir,'samstimRun.bat')

        with open(batFile,'w') as f:
            f.write(toRun)
            
        p = subprocess.Popen([batFile])
        p.wait()


if __name__=="__main__":
    start()