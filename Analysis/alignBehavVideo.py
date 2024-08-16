import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from DynamicRoutingAnalysisUtils import DynRoutData



baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

# please don't ever open these files in excel; make a local copy first if you want to look at them
drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

mice = ['728060','728916']

visStimOnsetDiff = []
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
                i = peak - c.size
                videoLickFrames = videoLickFrames[i-behavData.lickFrames.size:i]
            else:
                i = peak
                videoLickFrames = videoLickFrames[i:i+behavData.lickFrames.size]
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
        
        # save aligned frame times
        np.save(os.path.join(baseDir,'behaviorvideos_frametimes',mouseId,os.path.basename(videoPath)[:-3]+'npy'),videoFrameTimes)

        
        # validation: compare predicted vis stim onset frames (after aligning by licks) to 
        # actual vis stim onset frames (estimated from thresholded video roi)
        
        # find video frames corresponding to visual stimulus onset times in behavior file
        visOnsetTimes = behavData.stimStartTimes[np.in1d(behavData.trialStim,('vis1','vis2'))]
        predictedVisOnsetFrames = np.searchsorted(videoFrameTimes,visOnsetTimes) # first video frame after vis stim onset


        # find visual stimulus onset frames in video by thresholding roi over stimulus location
        videoIn = cv2.VideoCapture(videoPath)
        
        # videoIn.get(cv2.CAP_PROP_FRAME_COUNT)
        # videoIn.set(cv2.CAP_PROP_POS_FRAMES,predictedVisOnsetFrames[0]+3)
        # isFrame,videoFrame = videoIn.read() 
        # plt.imshow(videoFrame,cmap='gray')
        
        stimRoiIntensity = []
        nonStimRoiIntensity = []
        videoIn.set(cv2.CAP_PROP_POS_FRAMES,1) # ignore first frame (header)
        while True:
            isFrame,videoFrame = videoIn.read()
            if isFrame:
                videoFrame = cv2.cvtColor(videoFrame,cv2.COLOR_BGR2GRAY)
                stimRoiIntensity.append(videoFrame[:60,:30].mean())
                # nonStimRoiIntensity.append(videoFrame[:30,70:130].mean())
            else:
                break
        videoIn.release()
        
        stimRoiIntensity = np.array(stimRoiIntensity)
        # nonStimRoiIntensity = np.array(nonStimRoiIntensity)
        assert(stimRoiIntensity.size == numVideoFrames)
        
        # find roi intensity changes
        m = np.median(stimRoiIntensity)
        thresh = 0.05 * m
        aboveThresh = (stimRoiIntensity < m - thresh) | (stimRoiIntensity > m + thresh)
        threshFrames = np.where(aboveThresh)[0]
        
        # find onsets; remove first and last (start and end of session)
        d = np.concatenate(([0],np.diff(threshFrames)))
        onsetFrames = threshFrames[d > 30][:-1]
        
        # remove timeouts
        visOnsetFrames = np.array([i for i in onsetFrames if aboveThresh[i:i+20].sum() < 19])
        
        # remove onset of non-completed trial if present
        if behavData.endsWithNonCompletedTrial and (visOnsetFrames.size - predictedVisOnsetFrames.size) == 1:
            visOnsetFrames = visOnsetFrames[:-1]
        
        assert(visOnsetFrames.size == predictedVisOnsetFrames.size)


        # get difference between predicted and actual vis stim onset frames
        onsetDiff = predictedVisOnsetFrames - visOnsetFrames
        print(mouseId,startTime,(onsetDiff.min(),np.median(onsetDiff),onsetDiff.max()))
        visStimOnsetDiff.append(onsetDiff)


# plot difference between predicted and actual vis stim onset frames        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
d = np.concatenate(visStimOnsetDiff)
ax.hist(d,bins=np.arange(d.min()-1.5,d.max()+2.5),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Predicted vis stim onset frame (from aligning by licks) minus\n actual vis stim onset frame (from thresholded video roi)')
ax.set_ylabel('Count')
plt.tight_layout()




