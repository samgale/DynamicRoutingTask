# -*- coding: utf-8 -*-
"""
GUI for initiating camstim or samstim scripts

@author: SVC_CCG
"""

from __future__ import division
import os
import subprocess
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

        self.galvoXLabel = QtWidgets.QLabel('Galvo X (-5-5 V):')
        self.galvoXLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.galvoXEdit = QtWidgets.QLineEdit('-0.2')
        self.galvoXEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.galvoXEdit.editingFinished.connect(self.setGalvoValue)
        
        self.galvoYLabel = QtWidgets.QLabel('Galvo Y (-5-5 V):')
        self.galvoYLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.galvoYEdit = QtWidgets.QLineEdit('-2')
        self.galvoYEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.galvoYEdit.editingFinished.connect(self.setGalvoValue)
        
        self.laserAmpLabel = QtWidgets.QLabel('Laser Amplitude (0-5 V):')
        self.laserAmpLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.laserAmpEdit = QtWidgets.QLineEdit('0')
        self.laserAmpEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.laserAmpEdit.editingFinished.connect(self.setLaserAmpValue)

        self.laserDurLabel = QtWidgets.QLabel('Laser Duration (s):')
        self.laserDurLabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.laserDurEdit = QtWidgets.QLineEdit('2')
        self.laserDurEdit.setAlignment(QtCore.Qt.AlignHCenter)
        self.laserDurEdit.editingFinished.connect(self.setLaserDurValue)
        
        self.applyValuesButton = QtWidgets.QPushButton('Apply Values')
        self.applyValuesButton.clicked.connect(self.startTask)

        self.mainWidget = QtWidgets.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayoutGridSpacing(self.mainLayout,winHeight,winWidth,6,2)
        self.mainLayout.addWidget(self.rigNameLabel,0,0,1,1)
        self.mainLayout.addWidget(self.rigNameEdit,0,1,1,1)
        self.mainLayout.addWidget(self.galvoXLabel,1,0,1,1)
        self.mainLayout.addWidget(self.galvoXEdit,1,1,1,1)
        self.mainLayout.addWidget(self.galvoYLabel,2,0,1,1)
        self.mainLayout.addWidget(self.galvoYEdit,2,1,1,1)
        self.mainLayout.addWidget(self.laserAmpLabel,3,0,1,1)
        self.mainLayout.addWidget(self.laserAmpEdit,3,1,1,1)
        self.mainLayout.addWidget(self.laserDurLabel,4,0,1,1)
        self.mainLayout.addWidget(self.laserDurEdit,4,1,1,1)
        self.mainLayout.addWidget(self.applyValuesButton,5,0,1,2)
        self.mainWidget.setLayout(self.mainLayout)
        
        self.mainWin.show()

    def setLayoutGridSpacing(self,layout,height,width,rows,cols):
        for row in range(rows):
            layout.setRowMinimumHeight(row,int(height/rows))
            layout.setRowStretch(row,1)
        for col in range(cols):
            layout.setColumnMinimumWidth(col,int(width/cols))
            layout.setColumnStretch(col,1)

    def setGalvoValue(self):
        sender = self.mainWin.sender()
        val = float(sender.text())
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
        galvoX = self.galvoXEdit.text()
        galvoY = self.galvoYEdit.text()
        optoAmp = self.laserAmpEdit.text()
        optoDur = self.laserDurEdit.text()
        batString = ('python ' + '"' + scriptPath +'"' + 
                     ' --rigName ' + '"' + rigName + '"' + 
                     ' --taskScript ' + '"' + taskScript + '"' + 
                     ' --taskVersion ' + '"' + taskVersion + '"' +
                     ' --galvoX ' + galvoX +
                     ' --galvoY ' + galvoY +
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