# -*- coding: utf-8 -*-
"""
GUI for initiating camstim or samstim scripts

@author: SVC_CCG
"""

from __future__ import division
import os
import subprocess
from PyQt5 import QtCore, QtGui, QtWidgets


def start():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    obj = SamStimGui(app)
    app.exec_()


class SamStimGui():
    
    def __init__(self,app):
        self.app = app
        self.baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask"
        
        # main window
        winHeight = 600
        winWidth = 400
        self.mainWin = QtWidgets.QMainWindow()
        self.mainWin.setWindowTitle('SamStimGui')
        self.mainWin.resize(winWidth,winHeight)
        screenCenter = QtWidgets.QDesktopWidget().availableGeometry().center()
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(screenCenter)
        self.mainWin.move(mainWinRect.topLeft())

        # rig layouts
        self.rigLayout = []
        self.camstimButton = []
        self.samstimButton = []
        self.stimModeLayout = []
        self.stimModeGroupBox = []
        self.lightButton = []
        self.mouseIDLabel = []
        self.mouseIDEdit = []
        self.taskScriptLabel = []
        self.taskScriptEdit = []
        self.taskVersionLabel = []
        self.taskVersionEdit = []
        self.startTaskButton = []
        for _ in range(6):
            self.rigLayout.append(QtWidgets.QGridLayout())
            self.setLayoutGridSpacing(self.rigLayout[-1],winHeight/3,winWidth/2,6,3)

            self.camstimButton.append(QtWidgets.QRadioButton('camstim'))
            self.samstimButton.append(QtWidgets.QRadioButton('samstim'))
            self.samstimButton[-1].setChecked(True)
            self.stimModeLayout.append(QtWidgets.QHBoxLayout())
            for button in (self.camstimButton[-1],self.samstimButton[-1]):
                button.clicked.connect(self.setStimMode)
                self.stimModeLayout[-1].addWidget(button)
            self.stimModeGroupBox.append(QtWidgets.QGroupBox())
            self.stimModeGroupBox[-1].setLayout(self.stimModeLayout[-1])

            self.lightButton.append(QtWidgets.QPushButton('Light',checkable=True))
            self.lightButton[-1].clicked.connect(self.setLight)

            self.mouseIDLabel.append(QtWidgets.QLabel('Mouse ID:'))
            self.mouseIDLabel[-1].setAlignment(QtCore.Qt.AlignHCenter)
            self.mouseIDEdit.append(QtWidgets.QLineEdit())
            self.mouseIDEdit[-1].setAlignment(QtCore.Qt.AlignHCenter)

            self.taskScriptLabel.append(QtWidgets.QLabel('Task Script:'))
            self.taskScriptLabel[-1].setAlignment(QtCore.Qt.AlignHCenter)
            self.taskScriptEdit.append(QtWidgets.QLineEdit())
            self.taskScriptEdit[-1].setAlignment(QtCore.Qt.AlignHCenter)

            self.taskVersionLabel.append(QtWidgets.QLabel('Task Version:'))
            self.taskVersionLabel[-1].setAlignment(QtCore.Qt.AlignHCenter)
            self.taskVersionEdit.append(QtWidgets.QLineEdit())
            self.taskVersionEdit[-1].setAlignment(QtCore.Qt.AlignHCenter)

            self.startTaskButton.append(QtWidgets.QPushButton('Start Task'))
            self.startTaskButton[-1].clicked.connect(self.startTask)
            
            self.rigLayout[-1].addWidget(self.stimModeGroupBox[-1],0,1,1,1)
            self.rigLayout[-1].addWidget(self.lightButton[-1],1,1,1,1)
            self.rigLayout[-1].addWidget(self.mouseIDLabel[-1],2,0,1,1)
            self.rigLayout[-1].addWidget(self.mouseIDEdit[-1],2,1,1,2)
            self.rigLayout[-1].addWidget(self.taskScriptLabel[-1],3,0,1,1)
            self.rigLayout[-1].addWidget(self.taskScriptEdit[-1],3,1,1,2)
            self.rigLayout[-1].addWidget(self.taskVersionLabel[-1],4,0,1,1)
            self.rigLayout[-1].addWidget(self.taskVersionEdit[-1],4,1,1,2)
            self.rigLayout[-1].addWidget(self.startTaskButton[-1],5,1,1,1)

        # main layout
        self.mainWidget = QtWidgets.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayoutGridSpacing(self.mainLayout,winHeight,winWidth,3,2)
        self.mainWidget.setLayout(self.mainLayout)
        
        for n,layout in enumerate(self.rigLayout):
            i,j = (n,0) if n<3 else (n-3,1)
            self.mainLayout.addLayout(layout,i,j,1,1)
        
        self.mainWin.show()

    def setLayoutGridSpacing(self,layout,height,width,rows,cols):
        for row in range(rows):
            layout.setRowMinimumHeight(row,height/rows)
            layout.setRowStretch(row,1)
        for col in range(cols):
            layout.setColumnMinimumWidth(col,width/cols)
            layout.setColumnStretch(col,1)

    def setStimMode(self):
        sender = self.mainWin.sender()
        isSamstim = sender in self.samstimButton
        rig = self.samstimButton.index(sender) if isSamstim else self.camstimButton.index(sender)
        self.taskScriptEdit[rig].setEnabled(isSamstim)
        self.taskVersionEdit[rig].setEnabled(isSamstim)
        if isSamstim and self.lightButton[rig].isChecked():
            self.setLight()

    def setLight(self):
        sender = self.mainWin.sender()
        rig = self.rigLayout.index(sender)
        lightOn = not self.lightButton.isChecked()
        if self.camstimButton[rig].isChecked():
            scriptPath = os.path.join(self.baseDir,'camstimLight.py')
            batString = 'python' + '"' + scriptPath +'" ' + "--rigName" + '"E' + str(rig+1) + '" ' + '--lightOn' + str(lightOn)
            self.lightButton[rig].setChecked(not lightOn)
        elif lightOn:
            scriptPath = os.path.join(self.baseDir,'startTask.py')
            taskScript = os.path.join(self.baseDir,'TaskControl.py')
            batString = 'python' + '"' + scriptPath +'" ' + "--rigName" + '"E' + str(rig+1) + '" ' + '--taskScript' + '"' + taskScript + '"'
        self.runBatFile(batString)

    def startTask(self):
        pass

    def runBatFile(self,batString):
        anacondaActivatePath = r"C:\Users\svc_ncbehavior\Anaconda3\Scripts\activate.bat"
        anacondaPath = r"C:\Users\svc_ncbehavior\Anaconda3"

        toRun = ('call ' + anacondaActivatePath + anacondaPath + '\n' +
                 'call activate zro27' + '\n' +
                 batString)

        batFile = os.path.join(self.baseDir,'samstimRun.bat')

        with open(batFile,'w') as f:
            f.write(toRun)
            
        p = subprocess.Popen([batFile])
        p.wait()

if __name__=="__main__":
    start()