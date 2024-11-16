# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:50:25 2024

@author: svc_ccg
"""

import pathlib

baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask')

for fileName in ('DynamicRoutingTrainingNSB.xlsx','DynamicRoutingTraining.xlsx'):
    excelPath = os.path.join(baseDir,fileName)
    sheets = pd.read_excel(excelPath,sheet_name=None)
    writer =  pd.ExcelWriter(excelPath,mode='a',engine='openpyxl',if_sheet_exists='replace',datetime_format='%Y%m%d_%H%M%S')
    for mouseId in sheets.keys():
        if mouseId in ('all mice','dead'):
            continue
        df = sheets[mouseId]
        df.insert(df.shape[1],'muscimol',0)
        df.to_excel(writer,sheet_name=mouseId,index=False)
        sheet = writer.sheets[mouseId]
        
        if 'NSB' in fileName:
            for col in ('ABCDEFGHIJK'):
                if col in ('I','J','K'):
                    w = 10
                elif col in ('B','C','H'):
                    w = 15
                elif col=='D':
                    w = 40
                else:
                    w = 30
                sheet.column_dimensions[col].width = w
        else:    
            for col in ('ABCDEFGHIJK'):
                if col in ('H','I','J','K'):
                    w = 10
                elif col in ('B','G'):
                    w = 15
                elif col=='C':
                    w = 40
                else:
                    w = 30
                sheet.column_dimensions[col].width = w
    
    writer.save()
    writer.close()



