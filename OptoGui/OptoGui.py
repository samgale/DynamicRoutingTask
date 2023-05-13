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
from DynamicRoutingOptoParams import getBregmaGalvoCalbirationData, galvoToBregma, bregmaToGalvo



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
        
        winHeight = 150
        winWidth = 150
        self.mainWin = QtWidgets.QMainWindow()
        self.mainWin.setWindowTitle('OptoGui')
        self.mainWin.resize(winWidth,winHeight)
        screenCenter = QtWidgets.QDesktopWidget().availableGeometry().center()
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(screenCenter)
        self.mainWin.move(mainWinRect.topLeft())
        
        self.rigNameLabel = QtWidgets.QLabel('Rig Name:')
        self.rigNameLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.rigNameEdit = QtWidgets.QLineEdit('NP3')
        self.rigNameEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.rigNameEdit.editingFinished.connect(self.updateBregmaGalvoCalibrationData)
        
        self.galvoButton = QtWidgets.QRadioButton('Galvo (V)')
        self.bregmaButton = QtWidgets.QRadioButton('Bregma (mm)')
        self.galvoButton.setChecked(True)
        self.galvoMode = True
        self.modeLayout = QtWidgets.QHBoxLayout()
        for button in (self.galvoButton,self.bregmaButton):
            button.clicked.connect(self.setMode)
            self.modeLayout.addWidget(button)
        self.modeGroupBox = QtWidgets.QGroupBox()
        self.modeGroupBox.setLayout(self.modeLayout)

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
        
        self.laserAmpLabel = QtWidgets.QLabel('Laser Amplitude (0-5 V):')
        self.laserAmpLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.laserAmpEdit = QtWidgets.QLineEdit('0.4')
        self.laserAmpEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.laserAmpEdit.editingFinished.connect(self.setLaserAmpValue)

        self.laserDurLabel = QtWidgets.QLabel('Laser Duration (s):')
        self.laserDurLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.laserDurEdit = QtWidgets.QLineEdit('1')
        self.laserDurEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.laserDurEdit.editingFinished.connect(self.setLaserDurValue)
        
        self.applyValuesButton = QtWidgets.QPushButton('Apply Values')
        self.applyValuesButton.clicked.connect(self.startTask)

        self.mainWidget = QtWidgets.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayoutGridSpacing(self.mainLayout,winHeight,winWidth,7,2)
        self.mainLayout.addWidget(self.rigNameLabel,0,0,1,1)
        self.mainLayout.addWidget(self.rigNameEdit,0,1,1,1)
        self.mainLayout.addWidget(self.modeGroupBox,1,0,1,2)
        self.mainLayout.addWidget(self.xLabel,2,0,1,1)
        self.mainLayout.addWidget(self.xEdit,2,1,1,1)
        self.mainLayout.addWidget(self.yLabel,3,0,1,1)
        self.mainLayout.addWidget(self.yEdit,3,1,1,1)
        self.mainLayout.addWidget(self.laserAmpLabel,4,0,1,1)
        self.mainLayout.addWidget(self.laserAmpEdit,4,1,1,1)
        self.mainLayout.addWidget(self.laserDurLabel,5,0,1,1)
        self.mainLayout.addWidget(self.laserDurEdit,5,1,1,1)
        self.mainLayout.addWidget(self.applyValuesButton,6,0,1,2)
        self.mainWidget.setLayout(self.mainLayout)
        
        self.mainWin.show()

        self.updateBregmaGalvoCalibrationData()

    def setLayoutGridSpacing(self,layout,height,width,rows,cols):
        for row in range(rows):
            layout.setRowMinimumHeight(row,int(height/rows))
            layout.setRowStretch(row,1)
        for col in range(cols):
            layout.setColumnMinimumWidth(col,int(width/cols))
            layout.setColumnStretch(col,1)

    def updateBregmaGalvoCalibrationData(self):
        self.bregmaGalvoCalibrationData = getBregmaGalvoCalbirationData(self.rigNameEdit.text())
            
    def setMode(self):
        sender = self.mainWin.sender()
        if (sender==self.galvoButton and not self.galvoMode) or (sender==self.bregmaButton and self.galvoMode):
            self.galvoMode = not self.galvoMode
            x = float(self.xEdit.text())
            y = float(self.yEdit.text())
            x,y = bregmaToGalvo(self.bregmaGalvoCalibrationData,x,y) if self.galvoMode else galvoToBregma(self.bregmaGalvoCalibrationData,x,y)
            self.xEdit.setText(str(round(x,3)))
            self.yEdit.setText(str(round(y,3)))

    def setXYValue(self):
        sender = self.mainWin.sender()
        val = float(sender.text())
        if self.galvoMode:
            if val < -5:
                sender.setText('-5')
            elif val > 5:
                sender.setText('5')
        else:
            d = self.bregmaGalvoCalibrationData['bregmaX'] if sender==self.xEdit else self.bregmaGalvoCalibrationData['bregmaY']
            low = d.min()
            high = d.max()
            if val < low:
                sender.setText(str(low))
            elif val > high:
                sender.setText(str(high))
    
    def setLaserAmpValue(self):
        val = float(self.laserAmpEdit.text())
        if val < 0:
            self.laserAmpEdit.setText('0')
        elif val > 5:
            self.laserAmpEdit.setText('5')

    def setLaserDurValue(self):
        val = float(self.laserDurEdit.text())
        if val < 0:
            self.laserDurEdit.setText('0')

    def startTask(self):
        rigName = self.rigNameEdit.text()
        scriptPath = os.path.join(self.baseDir,'OptoGui','startOptoTask.py')
        taskScript = os.path.join(self.baseDir,'TaskControl.py')
        taskVersion = 'opto test'
        x = self.xEdit.text()
        y = self.yEdit.text()
        if not self.galvoMode:
            x,y = [str(n) for n in bregmaToGalvo(self.bregmaGalvoCalibrationData,float(x),float(y))]
        optoAmp = self.laserAmpEdit.text()
        optoDur = self.laserDurEdit.text()
        batString = ('python ' + '"' + scriptPath +'"' + 
                     ' --rigName ' + '"' + rigName + '"' + 
                     ' --taskScript ' + '"' + taskScript + '"' + 
                     ' --taskVersion ' + '"' + taskVersion + '"' +
                     ' --galvoX ' + x +
                     ' --galvoY ' + y +
                     ' --optoAmp ' + optoAmp +
                     ' --optoDur ' + optoDur)
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