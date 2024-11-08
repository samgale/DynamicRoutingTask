#%%
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn
from sklearn.linear_model import LogisticRegression
import npc_lims
import npc_sessions


#%%
def getAlignedSpikes(spikeTimes,startTimes,windowDur,binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    spikes = np.zeros((len(startTimes),bins.size-1))
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikeTimes,start)
        endInd = np.searchsorted(spikeTimes,start+windowDur)
        spikes[i] = np.histogram(spikeTimes[startInd:endInd]-start, bins)[0]
    return spikes


def standardizeData(d):
    d -= d.mean(axis=0)
    s = d.std(axis=0)
    d[:,s>0] /= s[s>0]


def getTrainTestSplits(y,nSplits=5):
    # cross validation using stratified shuffle split
    # each split preserves the percentage of samples of each class
    # all samples used in one test set
    classVals = np.unique(y)
    nClasses = len(classVals)
    nSamples = y.size
    samplesPerClass = [np.sum(y==val) for val in classVals]

    samplesPerSplit = [round(n/nSplits) for n in samplesPerClass]
    shuffleInd = np.random.permutation(nSamples)
    trainInd = []
    testInd = []
    for k in range(nSplits):
        testInd.append([])
        for val,n in zip(classVals,samplesPerSplit):
            start = k*n
            ind = shuffleInd[y[shuffleInd]==val] 
            testInd[-1].extend(ind[start:start+n] if k+1<nSplits else ind[start:])
        trainInd.append(np.setdiff1d(shuffleInd,testInd[-1]))
    return trainInd,testInd


def runDecoder(X,y,trainTrials,testTrials,nShuffles):
    predict = np.zeros(y.size)
    prob = predict.copy()
    predictShuffled = np.zeros((nShuffles,y.size))
    for train,test in zip(trainTrials,testTrials):
        model = LogisticRegression(C=1.0,max_iter=int(1e4),class_weight='balanced')
        model.fit(X[train],y[train])
        predict[test] = model.predict(X[test])
        prob[test] = model.predict_proba(X[test])[:,1]
        for i in range(nShuffles):
            model.fit(X[train],np.random.permutation(y[train]))
            predictShuffled[i,test] = model.predict(X[test])
    accuracy = sklearn.metrics.balanced_accuracy_score(y,predict)
    accuracyShuffled = [sklearn.metrics.balanced_accuracy_score(y,p) for p in predictShuffled]
    return predict,prob,accuracy,accuracyShuffled


#%%
baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

miceToIgnore = summaryDf['wheel fixed'] | summaryDf['cannula']

hasIndirectRegimen = np.array(summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var'])

ind = ~hasIndirectRegimen & summaryDf['stage 5 pass'] & summaryDf['moving grating'] & summaryDf['AM noise'] & ~summaryDf['stage 5 repeats'] & ~miceToIgnore
mice = tuple(summaryDf[ind]['mouse id'])

nonStandardTrainingMice = (644864,644866,644867,681532,686176)
mice += nonStandardTrainingMice


# %%
nUnitSamples = 20
unitSampleSize = 20
windowDur = 1
binSize = windowDur
bins = np.arange(-windowDur,windowDur+binSize,binSize)
nBins = bins.size - 1
nShuffles = 100

sessionInfo = npc_lims.get_session_info()
results = {}
failedToLoad = []
for mi,mouse in enumerate(mice):
    print(str(mi+1)+'/'+str(len(mice)))
    mouse = str(mouse)
    sessions = [s for s in sessionInfo if s.subject==mouse and s.is_ephys]
    sessionIds = [s.id for s in sessions if s.is_annotated and len(s.issues)==0]
    for sid in sessionIds:
        try:
            session = npc_sessions.DynamicRoutingSession(sid)
            trialsDf = session.trials[:]
            unitsDf = session.units[:]
        except:
            failedToLoad.append((mouse,sid))
            continue

        trials = trialsDf.is_target & trialsDf.is_nogo & ~trialsDf.is_reward_scheduled
        trainTrials = []
        testTrials = []
        for blockInd in range(6):
            blockTrials = trialsDf.block_index == blockInd
            contextTrials = trialsDf.is_vis_context if trialsDf.is_vis_context[blockTrials].any() else trialsDf.is_aud_context
            trainTrials.append((contextTrials & ~blockTrials)[trials])
            testTrials.append(blockTrials[trials])
        startTimes = trialsDf.quiescent_stop_time[trials]
        y = np.array(trialsDf.is_response[trials])
        for structure in unitsDf.structure.unique():
            units = np.where(unitsDf.structure==structure)[0]
            if len(units) < unitSampleSize:
                continue
            if mouse not in results:
                results[mouse] = {}
            if sid not in results[mouse]:
                results[mouse][sid] = {'trials': trials}
            if structure not in results[mouse][sid]:
                results[mouse][sid][structure] = {}
            predict = np.zeros((nUnitSamples,nBins,y.size))
            prob = np.zeros((nUnitSamples,nBins,y.size))
            accuracy = np.zeros((nUnitSamples,nBins))
            accuracyShuffled = np.zeros((nUnitSamples,nBins,nShuffles))
            for i in range(nUnitSamples):
                unitSample = np.random.choice(units,unitSampleSize,replace=False)
                for j,(binStart,binEnd) in enumerate(zip(bins[:-1],bins[1:])):
                    X = np.zeros((len(startTimes),unitSampleSize))
                    for k,u in enumerate(unitSample):
                        spikeTimes = np.array(unitsDf.iloc[u].spike_times)
                        X[:,k] = getAlignedSpikes(spikeTimes,startTimes-binStart,binEnd-binStart,binSize)[:,0]
                    standardizeData(X)
                    predict[i,j],prob[i,j],accuracy[i,j],accuracyShuffled[i,j] = runDecoder(X,y,trainTrials,testTrials,nShuffles)
            
            r = results[mouse][sid][structure]
            r['prob'] = np.mean(prob,axis=0)
            r['accuracy'] = np.mean(accuracy,axis=0)
            r['accuracyShuffled'] = np.mean(np.percentile(accuracyShuffled,95,axis=-1),axis=0)
                        
                        



# %%
