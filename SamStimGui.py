# -*- coding: utf-8 -*-
"""
GUI for initiating camstim or samstim scripts

@author: SVC_CCG
"""

from __future__ import division
import os
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
        
        # main window
        winHeight = 200
        winWidth = 200
        self.mainWin = QtWidgets.QMainWindow()
        self.mainWin.setWindowTitle('SamStimGui')
        self.mainWin.resize(winWidth,winHeight)
        screenCenter = QtWidgets.QDesktopWidget().availableGeometry().center()
        mainWinRect = self.mainWin.frameGeometry()
        mainWinRect.moveCenter(screenCenter)
        self.mainWin.move(mainWinRect.topLeft())

        # rig layouts
        self.rigLayouts = []
        for _ in range(6):
            self.rigLayouts.append(QtWidgets.QGridLayout())
            self.setLayoutGridSpacing(self.mainLayout,winHeight/6,winWidth/6,5,1)
            
            self.useMtrainCheckbox = QtWidgets.QCheckBox('use mtrain')
            self.useMtrainCheckbox.clicked.connect(self.setUseMtrain)
            
            self.rigLayout[-1].addWidget(self.useMtrainCheckbox,0,0,1,1)

        # main layout
        self.mainWidget = QtWidgets.QWidget()
        self.mainWin.setCentralWidget(self.mainWidget)
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayoutGridSpacing(self.mainLayout,winHeight,winWidth,6,2)
        self.mainWidget.setLayout(self.mainLayout)
        
        for i,layout in enumerate(self.rigLayouts):
            self.mainLayout.addLayout(layout,0,0,1,1)
        
        self.mainWin.show()

    def setLayoutGridSpacing(self,layout,height,width,rows,cols):
        for row in range(rows):
            layout.setRowMinimumHeight(row,height/rows)
            layout.setRowStretch(row,1)
        for col in range(cols):
            layout.setColumnMinimumWidth(col,width/cols)
            layout.setColumnStretch(col,1)

    def setUseMtrain(self):
        pass
        

if __name__=="__main__":
    start()