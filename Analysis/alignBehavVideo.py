import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from DynamicRoutingAnalysisUtils import DynRoutData



baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

# don't open these files in excel; make a local copy first
drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

mice = ['728060','728916']
roi = [38:63,22:36]

for mouseId in mice:
    df = drSheets[mouseId] if mouseId in drSheets else nsbSheets[mouseId]
    sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
    for startTime in df.loc[sessions]['start time']:
        startTime = startTime.strftime('%Y%m%d_%H%M%S')
        
        # check if video and video data files exist
        videoPath = glob.glob(os.path.join(baseDir,'behaviorvideos',mouseId,mouseId+'*'+startTime[:8]+'*.mp4'))
        videoDataPath = glob.glob(os.path.join(baseDir,'behaviorvideos',mouseId,mouseId+'*'+startTime[:8]+'*.json'))
        if len(videoPath) != 1 or len(videoDataPath) != 1:
            continue
        videoPath = videoPath[0]
        videoDataPath = videoDataPath[0]
        
        
        # get video data
        with open(videoDataPath,'r') as f:
            videoData = json.load(f)['RecordingReport']

        lostFrames = videoData['LostFrames']
        assert(len(lostFrames) < 1) # haven't found any videos with lost frames, so not dealing with them yet

        numVideoFrames = videoData['FramesRecorded']
        videoFrameRate = videoData['FPS']

        cameraInput = np.array([int(v) for v in videoData['CameraInput'][0].split(',')])
        videoLickFrames = cameraInput[0::2][cameraInput[1::2].astype(bool)]
        assert(videoLickFrames.size == videoData['CameraInputCount'])
        
        
        # get behavior data
        behavFileName = 'DynamicRouting1_' + mouseId + '_' + startTime + '.hdf5'
        behavFilePath = os.path.join(baseDir,'DynamicRoutingTask','Data',mouseId,behavFileName)
        behavData = DynRoutData()
        behavData.loadBehavData(behavFilePath)
        
        
        # remove video licks before or after behavior session
        if videoLickFrames.size > behavData.lickFrames.size:
            videoLickIntervals = np.diff(videoLickFrames) * 2
            behavLickIntervals = np.diff(behavData.lickFrames)
            c = np.correlate(videoLickIntervals,behavLickIntervals)
            peak = np.argmax(c)
            if peak > videoLickIntervals.size:
                videoLickFrames = videoLickFrames[:peak-c.size]
            else:
                videoLickFrames = videoLickFrames[peak:]
        
        assert(videoLickFrames.size == behavData.lickFrames.size)
        
        
        # get video frame times aligned to behavior session
        videoFrameTimes = [] # list of arrays of frame times between licks
        
        # get frame times to first lick using time of first lick and video frame rate
        videoStartTime = behavData.lickTimes[0] - videoLickFrames[0]/videoFrameRate # relative to start of behavior session
        videoFrameTimes.append(np.linspace(videoStartTime, behavData.lickTimes[0], videoLickFrames[0])[:-1])
        
        # get frame times between licks using frame interval implied by interval between licks and number of video frames
        for i in range(videoLickFrames.size-1):
            nFrames = videoLickFrames[i+1] - videoLickFrames[i]
            videoFrameTimes.append(np.linspace(behavData.lickTimes[i], behavData.lickTimes[i+1], nFrames + 1)[:-1])
            
        # get frame times after last lick
        nFrames = numVideoFrames - videoLickFrames[-1]
        videoStopTime = behavData.lickTimes[-1] + nFrames/videoFrameRate
        videoFrameTimes.append(np.linspace(behavData.lickTimes[-1], videoStopTime, nFrames + 1))
        
        # concatenate all the frame times
        videoFrameTimes = np.concatenate(videoFrameTimes)
        assert(videoFrameTimes.size == numVideoFrames)


        # find video frames corresponding to visual stimulus onset times in behavior file
        visOnsetTimes = behavData.stimStartTimes[np.in1d(behavData.trialStim,('vis1','vis2'))]
        predictedVisOnsetFrames = np.searchsorted(videoFrameTimes,visOnsetTimes) # first video frame after vis stim onset


        # find visual stimulus onset frames in video by thresholding roi over stimulus location
        videoIn = cv2.VideoCapture(videoPath)
        
        # videoIn.get(cv2.CAP_PROP_FRAME_COUNT)
        # videoIn.set(cv2.CAP_PROP_POS_FRAMES,predictedVisOnsetFrames[0])
        # plt.imshow(videoFrame,cmap='gray')
        
        roiIntensity = []
        isFrame,videoFrame = videoIn.read() # ignore first frame (header)
        while True:
            isFrame,videoFrame = videoIn.read()
            if isFrame:
                videoFrame = cv2.cvtColor(videoFrame,cv2.COLOR_BGR2GRAY)
                roiIntensity.append(videoFrame[38:63,22:36].mean())
            else:
                videoIn.release()
                break
        roiIntensity = np.array(roiIntensity)
        assert(roiIntensity.size == numVideoFrames)

        # find vis onset frames; ignore first and last threshold crossing
        medianIntensity = np.median(roiIntensity)
        threshold = 0.05 * medianIntensity
        visOnsetFrames = np.where((roiIntensity < medianIntensity - threshold) | (roiIntensity > medianIntensity + threshold))[0]
        visOnsetFrames = visOnsetFrames[np.concatenate(([False],np.diff(visOnsetFrames) > 30))][:-1]
        assert(visOnsetFrames.size == predictedVisOnsetFrames.size)


        # plot offset between predicted and actual vis stim onset frames
        offset = predictedVisOnsetFrames - visOnsetFrames
        print(mouseId,startTime,(offset.min(),np.median(offset),offset.max()))
        
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # ax.hist(offset,bins=np.arange(-5,6))
        # for side in ('right','top'):
        #     ax.spines[side].set_visible(False)
        # ax.tick_params(direction='out',top=False,right=False)
        # ax.set_xlabel('Predicted vis stim onset frame (from aligning by licks) minus\n actual vis stim onset frame (from thresholded video roi)')
        # ax.set_ylabel('Count')
        # plt.tight_layout()




