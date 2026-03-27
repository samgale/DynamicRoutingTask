import argparse
import copy
import multiprocessing
import os
import pathlib
import random
import typing
import numpy as np
import pandas as pd
from DynamicRoutingAnalysisUtils import getIsStandardRegimen,getSessionsToPass,getSessionData


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting')


def getData():
    # get data for pooled training
    summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies','BehaviorSummary.xlsx'),sheet_name=None)
    summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))
    drSheets,nsbSheets = [pd.read_excel(os.path.join(baseDir,'Sam','behav_spreadsheet_copies',fileName),sheet_name=None) for fileName in ('DynamicRoutingTraining.xlsx','DynamicRoutingTrainingNSB.xlsx')]
    isStandardRegimen = getIsStandardRegimen(summaryDf)
    mice = np.array(summaryDf[isStandardRegimen & summaryDf['stage 5 pass'] ]['mouse id'])

    sessionDataByMouse = {phase: [] for phase in ('initial training','after learning')}
    for mouseId in mice:
        df = drSheets[str(mouseId)] if str(mouseId) in drSheets else nsbSheets[str(mouseId)]
        sessions = np.array(['stage 5' in task for task in df['task version']]) & ~np.array(df['ignore'].astype(bool))
        sessions = np.where(sessions)[0]
        sessionsToPass = getSessionsToPass(mouseId,df,sessions,stage=5)
        # sessionDataByMouse['initial training'].append([getSessionData(mouseId,startTime,lightLoad=True) for startTime in df.loc[sessions[:2],'start time']])
        sessionDataByMouse['after learning'].append([getSessionData(mouseId,startTime,lightLoad=True) for startTime in df.loc[sessions[sessionsToPass:sessionsToPass+2],'start time']])
        
    trainingPhase = 'after learning'
    sessionData = (# first session from odd mice, second session from even mice
                   [s for d in sessionDataByMouse[trainingPhase][::2] for s in d[0:1]] + [s for d in sessionDataByMouse[trainingPhase][1::2] for s in d[1:2]]
                   # second session from odd mice, first session from even mice
                   + [s for d in sessionDataByMouse[trainingPhase][::2] for s in d[1:2]] + [s for d in sessionDataByMouse[trainingPhase][1::2] for s in d[0:1]])
    testIndex = np.arange(len(mice))
    trainIndex = np.arange(len(mice),2*len(mice))

    return sessionData,testIndex,trainIndex


def trainModel(nProcesses,modelType,sessionData,testIndex,trainIndex,latentPenalty,updatePenalty,latentPenInd,updatePenInd):
    if nProcesses > 1:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(1/nProcesses)
    from disentangled_rnns.library import disrnn
    from disentangled_rnns.library import rnn_utils
    from disentangled_rnns.library import two_armed_bandits
    import haiku as hk
    import jax
    import optax

    # make disrnn datasets
    nUpdateInputs = 6
    nChoiceInputs = 4
    maxTrials = max(session.nTrials for session in sessionData)
    updateInput = -1 * np.ones((maxTrials,len(sessionData),nUpdateInputs),dtype=np.float32)
    choiceInput = -1 * np.ones((maxTrials,len(sessionData),nChoiceInputs),dtype=np.float32)
    targetOutput = -1 * np.ones((maxTrials,len(sessionData),1),dtype=np.int32)
    stimNames = ('vis1','vis2','sound1','sound2')
    for i,session in enumerate(sessionData):
        n = session.nTrials
        for j,stim in enumerate(stimNames):
            isStim = session.trialStim == stim
            updateInput[1:n,i,j] = isStim[:-1]
            choiceInput[:n,i,j] = isStim
        updateInput[0,i,:] = 0
        updateInput[1:n,i,4] = session.trialResponse[:-1]
        updateInput[1:n,i,5] = session.trialRewarded[:-1]
        targetOutput[:n,i,0] = session.trialResponse
    
    if modelType == 'gru':
        gruInput = updateInput.copy()
        gruInput[:,:,:nChoiceInputs] = choiceInput
        testDataset,trainDataset = [rnn_utils.DatasetRNN(
                xs=gruInput[:,i],
                xs_choice=None,
                ys=targetOutput[:,i],
                y_type='categorical',
                n_classes=2,
                x_names=['VIS+','VIS-','AUD+','AUD-','response','outcome'],
                y_names=['response'],
                batch_size=1,
                batch_mode='random') # random or rolling
            for i in (testIndex,trainIndex)]
    else:
        testDataset,trainDataset = [rnn_utils.DatasetRNN(
                xs=updateInput[:,i],
                xs_choice=choiceInput[:,i],
                ys=targetOutput[:,i],
                y_type='categorical',
                n_classes=2,
                x_names=['VIS+','VIS-','AUD+','AUD-','response','outcome'],
                xs_choice_names=['VIS+','VIS-','AUD+','AUD-'],
                y_names=['response'],
                batch_size=1,
                batch_mode='random') # random or rolling
            for i in (testIndex,trainIndex)]
    
    # define the rnn architecture
    if modelType == 'disrnn':
        disrnn_config = disrnn.DisRnnConfig(
            # Dataset related
            obs_size=nUpdateInputs,
            choice_obs_size=nChoiceInputs,
            output_size=2,
            x_names=testDataset.x_names,
            y_names=testDataset.y_names,
            # Network architecture
            latent_size=9,
            update_net_n_units_per_layer=16,
            update_net_n_layers=8,
            choice_net_n_units_per_layer=4,
            choice_net_n_layers=1,
            activation="leaky_relu",
            # Penalties
            noiseless_mode=False,
            latent_penalty=latentPenalty,
            update_net_obs_penalty=updatePenalty,
            update_net_latent_penalty=updatePenalty,
            choice_net_obs_penalty=0,
            choice_net_latent_penalty=latentPenalty,
                l2_scale=0.001)
        
        # Define a config for warmup training with no noise and no penalties
        disrnn_config_warmup = copy.deepcopy(disrnn_config)
        disrnn_config_warmup.latent_penalty = 0
        disrnn_config_warmup.choice_net_obs_penalty = 0
        disrnn_config_warmup.choice_net_latent_penalty = 0
        disrnn_config_warmup.update_net_obs_penalty = 0
        disrnn_config_warmup.update_net_latent_penalty = 0
        disrnn_config_warmup.l2_scale = 0
        disrnn_config_warmup.noiseless_mode = True
    
        # Define network builder functions
        make_disrnn = lambda: disrnn.HkDisentangledRNN(disrnn_config)
        make_disrnn_warmup = lambda: disrnn.HkDisentangledRNN(disrnn_config_warmup)
        make_network = make_disrnn
        make_eval_network = make_disrnn_warmup
        loss = "penalized_categorical"
    else:
        make_gru = lambda: hk.DeepRNN([hk.GRU(8), hk.Linear(2)])
        make_network = make_gru
        make_eval_network = make_gru
        loss = "categorical"
    
    # Define an optimizer
    opt = optax.adam(learning_rate=0.001) 
    
    # Warmup training with no noise and no penalties
    if modelType == 'disrnn':
        params,_,_ = rnn_utils.train_network(
            make_disrnn_warmup,
            training_dataset=trainDataset,
            validation_dataset=testDataset,
            loss=loss,
            params=None,
            opt_state=None,
            opt=opt,
            n_steps=1000,
            report_progress_by='none',
            do_plot=False)
    else:
        params = None
    
    # Additional training using information penalty
    params,opt_state,losses = rnn_utils.train_network(
        make_network,
        training_dataset=trainDataset,
        validation_dataset=testDataset,
        loss=loss,
        params=params,
        opt_state=None,
        opt=opt,
        n_steps=10000,
        log_losses_every=10,
        report_progress_by='none',
        do_plot=False)
    
    # store model params
    modelParams = params
    modelLosses = losses
    if modelType == 'disrnn':
        modelConfig = disrnn_config
        latentSigmas = np.array(disrnn.reparameterize_sigma(params['hk_disentangled_rnn']['latent_sigma_params']))
        latentOrder = np.argsort(latentSigmas)
    else:
        modelConfig = None
        latentSigmas = None
        latentOrder = None
    
    # Eval network on unseen data
    # Use the wamrup disrnn so that there will be no noise
    for s in np.arange(len(testIndex)):
        xs = testDataset.get_all()['xs'][:,[s]]
        ys = testDataset._ys[:,[s]]
        network_outputs,network_states = rnn_utils.eval_network(make_eval_network,params,xs)
        latentStates = network_states[:,0]
        # probResp = np.exp(network_outputs[:,0,1]) / (np.exp(network_outputs[:,0,0]) + np.exp(network_outputs[:,0,1]))
        probResp = np.array(jax.nn.softmax(network_outputs[:,0,:2]))[:,1]
        likelihood = rnn_utils.normalized_likelihood(ys,network_outputs[:,:,:2])

    # simulate behavior with trained network
    class DynamicRoutingTaskEnv(two_armed_bandits.BaseEnvironment):
        def __init__(self,
                     networkInput: np.ndarray, # DatasetRNN._xs for one session (ntrials x 1 x ninputs) 
                     trialStim: np.ndarray,
                     rewardedStim: np.ndarray,
                     rewardScheduled: np.ndarray,
                     response = np.ndarray,
                     reward = np.ndarray,
                     seed: typing.Optional[int] = None,
                     n_arms: int = 2):
            super().__init__(seed=seed, n_arms=n_arms)
            self.networkInput = networkInput
            self.response = response
            self.reward = reward
            self.trialStim = trialStim
            self.rewardedStim = rewardedStim
            self.rewardScheduled = rewardScheduled
            self.new_session()
          
        def new_session(self):
            self.xs = self.networkInput.copy()
            self.probResp = []
          
        def step(self, attempted_choice: int, choice_probs: np.ndarray, trial_index: int):
            self.probResp.append(choice_probs[1])
            choice = attempted_choice
            instructed = self.rewardScheduled[trial_index]
            reward = (choice and self.trialStim[trial_index] == self.rewardedStim[trial_index]) or instructed
            # choice = self.response[trial_index]
            # reward = self.reward[trial_index]
            if trial_index < self.trialStim.size - 1:
                xs = self.xs[trial_index + 1]
                xs[0,4] = choice
                xs[0,5] = reward
            else:
                xs = None
            return choice, reward, xs

    simResp = []
    simProbResp = []
    agent = two_armed_bandits.AgentNetwork(make_eval_network,modelParams)
    for sessionInd,session in enumerate(np.array(sessionData)[testIndex]):
        networkInput = testDataset.get_all()['xs'][:,[sessionInd]]
        env = DynamicRoutingTaskEnv(networkInput,session.trialStim,session.rewardedStim,session.autoRewardScheduled,session.trialResponse,session.trialRewarded)
        simResp.append([])
        simProbResp.append([])
        for _ in range(10):
            d = two_armed_bandits.create_dataset(agent,env,session.nTrials,1)
            simResp[-1].append(d._ys[:,0,0])
            simProbResp[-1].append(np.array(env.probResp))

    # save data
    fileName = modelType+'_latPenInd'+str(latentPenInd)+'_updPenInd'+str(updatePenInd)+'.npz'
    filePath = os.path.join(baseDir,'Sam','DisRNNmodel',fileName)
    np.savez_compressed(filePath,modelParams=modelParams,modelLosses=modelLosses,modelConfig=modelConfig,latentSigmas=latentSigmas,latentOrder=latentOrder,
                        latentStates=latentStates,probResp=probResp,likelihood=likelihood,simResp=simResp,simProbResp=simProbResp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nProcesses',type=int)
    args = parser.parse_args()
    nProcesses = args.nProcesses

    sessionData,testIndex,trainIndex = getData()
    latentPenalties = [0.0003] #[0.01,0.003,0.001,0.0003,0.0001,0.00003,0.00001,0.000003]
    updatePenalties = [0.003,0.001] #[0.03,0.01,0.003,0.001,0.0003]

    poolArgs = []
    for modelType in ('gru','disrnn'):
        for latPenInd,latPen in enumerate(latentPenalties if modelType=='disrnn' else [None]):
            for updPenInd,updPen in enumerate(updatePenalties if modelType=='disrnn' else [None]):
                poolArgs.append((nProcesses,modelType,sessionData,testIndex,trainIndex,latPen,updPen,latPenInd,updPenInd))

    multiprocessing.set_start_method('spawn',force=True)
    with multiprocessing.Pool(processes=nProcesses) as pool:
        pool.starmap(trainModel,poolArgs)
