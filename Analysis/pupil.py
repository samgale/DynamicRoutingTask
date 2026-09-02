# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:03:08 2026

@author: samg
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import polars as pl
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getStage5Sessions,getSessionsToPass,getSessionData


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

isStandardRegimen = getIsStandardRegimen(summaryDf)

mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass']]['mouse id'])
sessionData = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = getStage5Sessions(mid,df)
    sessionsToPass = getSessionsToPass(mid,df,sessions,stage=5)
    sessionData.append([getSessionData(mid,startTime,lightLoad=True) for startTime in df.loc[sessions[sessionsToPass:],'start time']])


eyeDf = (
    pl.scan_parquet(
        "s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.289/consolidated/eye_tracking.parquet",
        storage_options={"skip_signature": "true"}, # public bucket, no credentials needed    
    )
    .select(
        "session_id", "subject_id", "pupil_area", "timestamps", "pupil_is_bad_frame",
    )
    .collect()
)


eyeMice = eyeDf['subject_id'].unique()


inEyeMice = [str(m) in eyeMice for m in mice]






# get on session's data, filter out low-confidence frames and apply a rolling median filter to remove outliers
med_filter_size = 10
df = (
    eye_df
    .filter(
        pl.col("session_id") == pl.col("session_id").first(), # or use .unique().sample() to get random session
        ~pl.col("pupil_is_bad_frame"),
    )
    .with_columns(
        pl.col("pupil_area").rolling_median(med_filter_size).alias("filtered_pupil_area"),
    )
    .sort("session_id", "timestamps")
)

fig, ax = plt.subplots()
ax.plot(df["timestamps"], df["pupil_area"], lw=.5)
ax.plot(df["timestamps"], df["filtered_pupil_area"], lw=.5)
ax.set_ylim(0, 20_000)
ax.set_xlabel("time (s)")
ax.set_ylabel("pupil area (pixels)")