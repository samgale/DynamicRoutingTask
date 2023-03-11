# -*- coding: utf-8 -*-
"""
GUI for initiating camstim or samstim scripts

@author: SVC_CCG
"""

from __future__ import division
import os
import subprocess
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets


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
        self.bregmaGalvoData = self.getBregmaGalvoData()
        
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
        self.laserAmpEdit = QtWidgets.QLineEdit('0.2')
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

    def setLayoutGridSpacing(self,layout,height,width,rows,cols):
        for row in range(rows):
            layout.setRowMinimumHeight(row,int(height/rows))
            layout.setRowStretch(row,1)
        for col in range(cols):
            layout.setColumnMinimumWidth(col,int(width/cols))
            layout.setColumnStretch(col,1)
            
    def setMode(self):
        sender = self.mainWin.sender()
        if (sender==self.galvoButton and not self.galvoMode) or (sender==self.bregmaButton and self.galvoMode):
            self.galvoMode = not self.galvoMode
            x = float(self.xEdit.text())
            y = float(self.yEdit.text())
            x,y = self.bregmaToGalvo(x,y) if self.galvoMode else self.galvoToBregma(x,y)
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
            x,y = [str(n) for n in self.bregmaToGalvo(float(x),float(y))]
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

    def getBregmaGalvoData(self):
        f = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\OptoGui\NP3_bregma_galvo.txt"
        d = pd.read_csv(f,sep='\t')
        
        bregmaToGalvoFit = np.linalg.lstsq(np.concatenate((d[['bregma x','bregma y']],np.ones(d.shape[0])[:,None]),axis=1),
                                           d[['galvo x','galvo y']])[0].T

        galvoToBregmaFit = np.linalg.lstsq(np.concatenate((d[['galvo x','galvo y']],np.ones(d.shape[0])[:,None]),axis=1),
                                           d[['bregma x','bregma y']])[0].T
        
        return bregmaToGalvoFit,galvoToBregmaFit

    def bregmaToGalvo(self,bregmaX,bregmaY):
        bregmaToGalvoFit = self.bregmaGalvoData[0]
        galvoX = bregmaToGalvoFit[0,0]*bregmaX + bregmaToGalvoFit[0,1]*bregmaY + bregmaToGalvoFit[0,2]
        galvoY = bregmaToGalvoFit[1,0]*bregmaX + bregmaToGalvoFit[1,1]*bregmaY + bregmaToGalvoFit[1,2]
        return galvoX,galvoY
    
    def galvoToBregma(self,galvoX,galvoY):
        galvoToBregmaFit = self.bregmaGalvoData[1]
        bregmaX = galvoToBregmaFit[0,0]*galvoX + galvoToBregmaFit[0,1]*galvoY + galvoToBregmaFit[0,2]
        bregmaY = galvoToBregmaFit[1,0]*galvoX + galvoToBregmaFit[1,1]*galvoY + galvoToBregmaFit[1,2]
        return bregmaX,bregmaY


if __name__=="__main__":
    start()