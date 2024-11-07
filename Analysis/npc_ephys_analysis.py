#%%
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


# %%
mouse = '702131'

sessionInfo = npc_lims.get_session_info()

sessions = [s for s in sessionInfo if s.subject==mouse and s.is_ephys]


# %%
session = npc_sessions.DynamicRoutingSession(sessions[-1].id)


#%%
trialsDf = session.trials[:]


# %%
unitsDf = session.units[:]


# %%
minUnits = 20
windowDur = 1
trialInd = (trialsDf.is_vis_context & trialsDf.is_aud_target) | (trialsDf.is_aud_context & trialsDf.is_vis_target)
startTimes = trialsDf.quiescent_stop_time[trialInd] - windowDur
y = np.array(~np.isnan(trialsDf.response_time[trialInd]))
for structure in unitsDf.structure.unique():
    if structure == 'out of brain':
        continue
    uind = np.where(unitsDf.structure==structure)[0]
    if len(uind) >= minUnits:
        X = np.zeros((len(startTimes),len(uind)))
        for i,u in enumerate(uind):
            spikeTimes = np.array(unitsDf.iloc[u].spike_times)
            X[:,i] = getAlignedSpikes(spikeTimes,startTimes,windowDur,windowDur)[:,0]
        m = X.mean(axis=0)
        X -= m
        X = X[:,m>0]
        X /= X.std(axis=0)
        # trainInd,testInd = getTrainTestSplits(y,nSplits=5)
        # predict = np.zeros(y.size,dtype=bool)
        # prob = np.zeros(y.size)
        # for train,test in zip(trainInd,testInd):
        #     #model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
        #     model = LogisticRegression(C=1.0,max_iter=int(1e4),class_weight='balanced')
        #     model.fit(X[train],y[train])
        #     predict[test] = model.predict(X[test])
        #     prob[test] = model.predict_proba(X[test])[:,1]
        # accuracy = sklearn.metrics.balanced_accuracy_score(y,predict)

        predict = np.zeros(y.size,dtype=bool)
        prob = np.zeros(y.size)
        for trial in range(y.size):
            model = LogisticRegression(C=1.0,max_iter=int(1e4),class_weight='balanced')
            trainInd = np.delete(np.arange(y.size),trial)
            model.fit(X[trainInd],y[trainInd])
            predict[trial] = model.predict(X[trial][None,:])[0]
            prob[trial] = model.predict_proba(X[trial][None,:])[0,1]
        accuracy = sklearn.metrics.balanced_accuracy_score(y,predict)
        print(structure,accuracy,len(uind))





# %%
