# -*- coding: utf-8 -*-
"""
GUI for initiating camstim or samstim scripts

@author: SVC_CCG
"""

from __future__ import division
import os
import subprocess
import pandas as pd
from PyQt5 import QtCore, QtWidgets


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
        self.githubPath = r"https://raw.githubusercontent.com/samgale/DynamicRoutingTask/32513f6cc852b0cc3fede05f89f2a6f6ad7e1c89"
        
        # main window
        winHeight = 600
        winWidth = 450
        self.mainWin = QtWidgets.QMainWindow()
        self.mainWin.setWindowTitle('SamStimGui')
        self.mainWin.resize(winWidth,winHeight)
        screenCenter = QtWidgets.QDesktopWidget().availableGeometry().center()
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(screenCenter)
        self.mainWin.move(mainWinRect.topLeft())

        self.clearAllButton = QtWidgets.QPushButton('Clear All')
        self.clearAllButton.clicked.connect(self.clearAll)

        self.updateAllButton = QtWidgets.QPushButton('Update All')
        self.updateAllButton.clicked.connect(self.updateAll)

        self.githubPathLabel = QtWidgets.QLabel('Github Path:')
        self.githubPathEdit = QtWidgets.QLineEdit(self.githubPath)

        # rig layouts
        self.rigGroupBox = []
        self.rigLayout = []
        self.userNameLabel = []
        self.userNameEdit = []
        self.camstimButton = []
        self.samstimButton = []
        self.stimModeLayout = []
        self.stimModeGroupBox = []
        self.useGithubCheckbox = []
        self.solenoidButton = []
        self.luminanceTestButton = []
        self.waterTestButton = []
        self.lightButton = []
        self.mouseIDLabel = []
        self.mouseIDEdit = []
        self.taskScriptLabel = []
        self.taskScriptEdit = []
        self.taskVersionLabel = []
        self.taskVersionEdit = []
        self.startTaskButton = []
        self.stopTaskButton = []
        for n in range(6):
            self.rigLayout.append(QtWidgets.QGridLayout())
            self.setLayoutGridSpacing(self.rigLayout[-1],winHeight/3,winWidth/2,8,4)

            self.userNameLabel.append(QtWidgets.QLabel('User Name:'))
            self.userNameLabel[-1].setAlignment(QtCore.Qt.AlignVCenter)
            self.userNameEdit.append(QtWidgets.QLineEdit('samg'))
            self.userNameEdit[-1].setAlignment(QtCore.Qt.AlignHCenter)

            self.camstimButton.append(QtWidgets.QRadioButton('camstim'))
            self.samstimButton.append(QtWidgets.QRadioButton('samstim'))
            self.samstimButton[-1].setChecked(True)
            self.stimModeLayout.append(QtWidgets.QHBoxLayout())
            for button in (self.camstimButton[-1],self.samstimButton[-1]):
                button.clicked.connect(self.setStimMode)
                self.stimModeLayout[-1].addWidget(button)
            self.stimModeGroupBox.append(QtWidgets.QGroupBox())
            self.stimModeGroupBox[-1].setLayout(self.stimModeLayout[-1])

            self.useGithubCheckbox.append(QtWidgets.QCheckBox('Get Script from Github'))
            self.useGithubCheckbox[-1].setChecked(True)
            
            self.solenoidButton.append(QtWidgets.QPushButton('Solenoid',checkable=True))
            self.solenoidButton[-1].clicked.connect(self.setSolenoid)

            self.waterTestButton.append(QtWidgets.QPushButton('Water Test'))
            self.waterTestButton[-1].clicked.connect(self.startWaterTest)

            self.lightButton.append(QtWidgets.QPushButton('Light'))
            self.lightButton[-1].clicked.connect(self.setLight)
            
            self.luminanceTestButton.append(QtWidgets.QPushButton('Luminance Test'))
            self.luminanceTestButton[-1].clicked.connect(self.startLuminanceTest)

            self.mouseIDLabel.append(QtWidgets.QLabel('Mouse ID:'))
            self.mouseIDLabel[-1].setAlignment(QtCore.Qt.AlignVCenter)
            self.mouseIDEdit.append(QtWidgets.QLineEdit())
            self.mouseIDEdit[-1].setAlignment(QtCore.Qt.AlignHCenter)
            self.mouseIDEdit[-1].editingFinished.connect(self.updateTask)

            self.taskScriptLabel.append(QtWidgets.QLabel('Task Script:'))
            self.taskScriptLabel[-1].setAlignment(QtCore.Qt.AlignVCenter)
            self.taskScriptEdit.append(QtWidgets.QLineEdit(''))
            self.taskScriptEdit[-1].setAlignment(QtCore.Qt.AlignHCenter)

            self.taskVersionLabel.append(QtWidgets.QLabel('Task Version:'))
            self.taskVersionLabel[-1].setAlignment(QtCore.Qt.AlignVCenter)
            self.taskVersionEdit.append(QtWidgets.QLineEdit(''))
            self.taskVersionEdit[-1].setAlignment(QtCore.Qt.AlignHCenter)

            self.startTaskButton.append(QtWidgets.QPushButton('Start Task'))
            self.startTaskButton[-1].clicked.connect(self.startTask)

            self.stopTaskButton.append(QtWidgets.QPushButton('Stop Task'))
            self.stopTaskButton[-1].clicked.connect(self.stopTask)
            self.stopTaskButton[-1].setEnabled(False)
            
            self.rigLayout[-1].addWidget(self.userNameLabel[-1],0,0,1,1)
            self.rigLayout[-1].addWidget(self.userNameEdit[-1],0,1,1,2)
            self.rigLayout[-1].addWidget(self.stimModeGroupBox[-1],1,0,1,2)
            self.rigLayout[-1].addWidget(self.useGithubCheckbox[-1],1,2,1,2)
            self.rigLayout[-1].addWidget(self.solenoidButton[-1],2,0,1,2)
            self.rigLayout[-1].addWidget(self.waterTestButton[-1],2,2,1,2)
            self.rigLayout[-1].addWidget(self.lightButton[-1],3,0,1,2)
            self.rigLayout[-1].addWidget(self.luminanceTestButton[-1],3,2,1,2)
            self.rigLayout[-1].addWidget(self.mouseIDLabel[-1],4,0,1,1)
            self.rigLayout[-1].addWidget(self.mouseIDEdit[-1],4,1,1,3)
            self.rigLayout[-1].addWidget(self.taskScriptLabel[-1],5,0,1,1)
            self.rigLayout[-1].addWidget(self.taskScriptEdit[-1],5,1,1,3)
            self.rigLayout[-1].addWidget(self.taskVersionLabel[-1],6,0,1,1)
            self.rigLayout[-1].addWidget(self.taskVersionEdit[-1],6,1,1,3)
            self.rigLayout[-1].addWidget(self.startTaskButton[-1],7,0,1,2)
            self.rigLayout[-1].addWidget(self.stopTaskButton[-1],7,2,1,2)
            
            self.rigGroupBox.append(QtWidgets.QGroupBox('B'+str(n+1)))
            self.rigGroupBox[-1].setLayout(self.rigLayout[-1])

        # main layout
        self.mainWidget = QtWidgets.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayoutGridSpacing(self.mainLayout,winHeight,winWidth,25,8)
        self.mainWidget.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.clearAllButton,0,0,1,2)
        self.mainLayout.addWidget(self.updateAllButton,0,2,1,2)
        self.mainLayout.addWidget(self.githubPathLabel,0,4,1,1)
        self.mainLayout.addWidget(self.githubPathEdit,0,5,1,3)
        for n,rigBox in enumerate(self.rigGroupBox):
            i,j = (n*8+1,0) if n<3 else ((n-3)*8+1,4)
            self.mainLayout.addWidget(rigBox,i,j,8,4)
        
        self.mainWin.show()

    def setLayoutGridSpacing(self,layout,height,width,rows,cols):
        for row in range(rows):
            layout.setRowMinimumHeight(row,height/rows)
            layout.setRowStretch(row,1)
        for col in range(cols):
            layout.setColumnMinimumWidth(col,width/cols)
            layout.setColumnStretch(col,1)

    def clearAll(self):
        for w in self.mouseIDEdit + self.taskScriptEdit + self.taskVersionEdit:
            w.setText('')

    def updateAll(self):
        self.loadTask()

    def setStimMode(self):
        sender = self.mainWin.sender()
        useSamstim = sender in self.samstimButton
        rig = self.samstimButton.index(sender) if useSamstim else self.camstimButton.index(sender)
        self.luminanceTestButton[rig].setEnabled(useSamstim)
        self.waterTestButton[rig].setEnabled(useSamstim)
        self.taskScriptEdit[rig].setEnabled(useSamstim)
        self.taskVersionEdit[rig].setEnabled(useSamstim)
        self.stopTaskButton[rig].setEnabled(not useSamstim)
        if useSamstim:
            if self.lightButton[rig].isChecked():
                self.setLight(False,rig=rig,camstim=True)
                self.lightButton[rig].setChecked(False)
        self.lightButton[rig].setCheckable(not useSamstim)
        
    def setSolenoid(self,checked):
        sender = self.mainWin.sender()
        rig = self.solenoidButton.index(sender)
        openSolenoid = checked
        if self.camstimButton[rig].isChecked():
            scriptPath = os.path.join(self.baseDir,'camstimControl.py')
            batString = ('python ' + '"' + scriptPath + '"' +
                         ' --rigName ' + '"B' + str(rig+1) + '"' +
                         ' --solenoidOpen ' + str(openSolenoid))
        else:
            scriptPath = os.path.join(self.baseDir,'startTask.py')
            taskScript = self.getTaskScript(rig=rig,scriptName='TaskControl')
            taskVersion = 'open solenoid' if openSolenoid else 'close solenoid'
            batString = ('python ' + '"' + scriptPath +'"' + 
                         ' --rigName ' + '"B' + str(rig+1) + '"' +  
                         ' --taskScript ' + '"' + taskScript + '"' + 
                         ' --taskVersion ' + '"' + taskVersion + '"')
        self.runBatFile(batString)

    def startWaterTest(self):
        sender = self.mainWin.sender()
        rig = self.waterTestButton.index(sender)
        scriptPath = os.path.join(self.baseDir,'startTask.py')
        taskScript = self.getTaskScript(rig=rig,scriptName='TaskControl')
        taskVersion = 'water test'
        batString = ('python ' + '"' + scriptPath +'"' + 
                     ' --rigName ' + '"B' + str(rig+1) + '"' +  
                     ' --taskScript ' + '"' + taskScript + '"' + 
                     ' --taskVersion ' + '"' + taskVersion + '"')
        self.runBatFile(batString)

    def setLight(self,checked,rig=None,camstim=None):
        if rig is None:
            sender = self.mainWin.sender()
            rig = self.lightButton.index(sender)
        if camstim is None:
            camstim = self.camstimButton[rig].isChecked()
        if camstim:
            scriptPath = os.path.join(self.baseDir,'camstimControl.py')
            batString = ('python ' + '"' + scriptPath + '"' +
                         ' --rigName ' + '"B' + str(rig+1) + '"' +
                         ' --lightOn ' + str(checked))
        else:
            scriptPath = os.path.join(self.baseDir,'startTask.py')
            taskScript = self.getTaskScript(rig=rig,scriptName='TaskControl')
            batString = ('python ' + '"' + scriptPath +'"' + 
                         ' --rigName ' + '"B' + str(rig+1) + '"' + 
                         ' --taskScript ' + '"' + taskScript + '"')
        self.runBatFile(batString)
    
    def startLuminanceTest(self):
        sender = self.mainWin.sender()
        rig = self.luminanceTestButton.index(sender)
        scriptPath = os.path.join(self.baseDir,'startTask.py')
        taskScript = self.getTaskScript(rig=rig,scriptName='TaskControl')
        taskVersion = 'luminance test'
        batString = ('python ' + '"' + scriptPath +'"' + 
                     ' --rigName ' + '"B' + str(rig+1) + '"' +  
                     ' --taskScript ' + '"' + taskScript + '"' + 
                     ' --taskVersion ' + '"' + taskVersion + '"')
        self.runBatFile(batString)

    def updateTask(self):
        sender = self.mainWin.sender()
        rig = self.mouseIDEdit.index(sender)
        self.loadTask([rig])
        
    def loadTask(self,rigs=None):
        if rigs is None:
            rigs = list(range(len(self.mouseIDEdit)))
        for rig in rigs:
            if self.samstimButton[rig].isChecked():
                mouseID = self.mouseIDEdit[rig].text()
                if mouseID == 'sound':
                    taskScript = 'TaskControl'
                    taskVersion = 'sound test'
                else:
                    try:
                        excelPath = os.path.join(self.baseDir,'DynamicRoutingTraining.xlsx')
                        df = pd.read_excel(excelPath,sheet_name='all mice')
                        row = df['mouse id'] == int(mouseID)
                        if row.sum() == 1:
                            taskScript = 'DynamicRouting1'
                            taskVersion = df[row]['next task version'].values[0]
                        else:
                            taskScript = ''
                            taskVersion = ''
                    except:
                        taskScript = ''
                        taskVersion = ''
                self.taskScriptEdit[rig].setText(taskScript)
                self.taskVersionEdit[rig].setText(taskVersion)
                
    def getTaskScript(self,rig=None,scriptName=None):
        if scriptName is None:
            scriptName = self.taskScriptEdit[rig].text()
        if self.useGithubCheckbox[rig].isChecked():
            taskScript = self.githubPathEdit.text() + '/' + scriptName + '.py'
        else:
            taskScript = os.path.join(self.baseDir,scriptName + '.py')
        return taskScript
        
    def startTask(self):
        sender = self.mainWin.sender()
        rig = self.startTaskButton.index(sender)
        userName = self.userNameEdit[rig].text()
        mouseID = self.mouseIDEdit[rig].text()
        if self.camstimButton[rig].isChecked():
            if len(mouseID) != 6:
                print('mouseID must be 6 digits')
                return
            scriptPath = os.path.join(self.baseDir,'camstimControl.py')
            batString = ('python ' + '"' + scriptPath +'"' + 
                         ' --rigName ' + '"B' + str(rig+1) + '"' +
                         ' --mouseID ' + '"' + mouseID + '"' +
                         ' --userName ' + '"' + userName + '"')
        else:
            scriptPath = os.path.join(self.baseDir,'startTask.py')
            taskScript = self.getTaskScript(rig=rig)
            taskVersion = self.taskVersionEdit[rig].text()
            batString = ('python ' + '"' + scriptPath +'"' + 
                         ' --userName ' + '"' + userName + '"' +
                         ' --rigName ' + '"B' + str(rig+1) + '"' + 
                         ' --subjectName ' + '"' + mouseID + '"' + 
                         ' --taskScript ' + '"' + taskScript + '"' + 
                         ' --taskVersion ' + '"' + taskVersion + '"')
        self.runBatFile(batString)

    def stopTask(self):
        sender = self.mainWin.sender()
        rig = self.startTaskButton.index(sender)
        if self.camstimButton[rig].isChecked():
            scriptPath = os.path.join(self.baseDir,'camstimControl.py')
            batString = ('python ' + '"' + scriptPath +'"' + 
                         ' --rigName ' + '"B' + str(rig+1) + '"')
            self.runBatFile(batString)

    def runBatFile(self,batString):
        toRun = ('call activate zro' + '\n' +
                 batString)

        batFile = os.path.join(self.baseDir,'samstimRun.bat')

        with open(batFile,'w') as f:
            f.write(toRun)
            
        p = subprocess.Popen([batFile])
        p.wait()

if __name__=="__main__":
    start()