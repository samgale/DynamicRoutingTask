# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:38:41 2025

@author: svc_ccg
"""

import random
import numpy as np
import torch
import torch.nn as nn
from DynamicRoutingAnalysisUtils import DynRoutData


filePath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data\818720\DynamicRouting1_818720_20251202_150802.hdf5"
sessionData = DynRoutData()
sessionData.loadBehavData(filePath,lightLoad=True)


nTrials = sessionData.nTrials
inputSize = 6
hiddenSize = 50
outputSize = 1

modelInput = np.zeros((nTrials,inputSize),dtype=np.float32)
for i,stim in enumerate(('vis1','vis2','sound1','sound2')):    
    modelInput[:,i] = sessionData.trialStim == stim
modelInput[1:,4] = sessionData.trialResponse[:-1]
modelInput[1:,5] = sessionData.trialRewarded[:-1]
modelInput = torch.from_numpy(modelInput)

targetOutput = torch.from_numpy(sessionData.trialResponse.astype(np.float32))


class CustomLSTM(nn.Module):
    def __init__(self,inputSize,hiddenSize,outputSize,sessionData,isSimulation=False):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTMCell(inputSize,hiddenSize,bias=True)
        self.linear = nn.Linear(hiddenSize,outputSize)
        self.sigmoid = nn.Sigmoid()
        self.sessionData = sessionData
        self.isSimulation = isSimulation

    def forward(self,inputSequence):
        pAction = []
        action = []
        reward = []
        for t in range(self.sessionData.nTrials):
            if self.isSimulation and t > 0:
                inputSequence[t,4] = action[t-1]
                inputSequence[t,5] = float(reward[t-1])
                
            h_t,c_t = self.lstm(inputSequence[t])
            output = self.linear(h_t)
            pAction.append(self.sigmoid(output)[0])
            if self.isSimulation:
                action.append(random.random() < pAction[-1])
                reward.append((action[-1] and self.sessionData.trialStim[t] == self.sessionData.rewardedStim[t]) or self.sessionData.autoRewardScheduled[t])
            else:
                action.append(self.sessionData.trialResponse[t])
                reward.append(self.sessionData.trialRewarded[t])
        
        return torch.stack(pAction),np.array(action),np.array(reward)



# model = CustomLSTM(inputSize,hiddenSize,outputSize,sessionData,isSimulation=False)
# pAction,action,reward = model(modelInput)
# pActionAsArray = pAction.detach().numpy()


model = CustomLSTM(inputSize,hiddenSize,outputSize,sessionData,isSimulation=False)
lossFunc = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(),lr=0.01)
nIters = 5
nFolds = 5
nTestTrials = round(nTrials / nFolds)
nEpochs = 30
logLossTrain = []
logLossTest = []
for _ in range(nIters):
    shuffleInd = np.random.permutation(nTrials)
    logLossTrain.append([])
    logLossTest.append([])
    for _ in range(nEpochs):
        modelOutput = model(modelInput)[0]
        loss = 0
        prediction = torch.zeros(nTrials,dtype=torch.float32)
        for k in range(nFolds):
            start = k * nTestTrials
            testTrials = shuffleInd[start:start+nTestTrials] if k+1 < nFolds else shuffleInd[start:]
            trainTrials = np.setdiff1d(shuffleInd,testTrials)
            loss += lossFunc(modelOutput[trainTrials],targetOutput[trainTrials])
            prediction[testTrials] = modelOutput[testTrials]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        logLossTrain[-1].append(loss.item() / nFolds)
        logLossTest[-1].append(lossFunc(prediction,targetOutput).item())   
        
        
        
    

# class CustomLSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(CustomLSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
#         self.linear = nn.Linear(hidden_size, output_size)

#     def forward(self, input_sequence, initial_hidden_state=None, initial_cell_state=None):
#         batch_size = input_sequence.size(0)
#         seq_len = input_sequence.size(1)

#         # Initialize hidden and cell states if not provided
#         if initial_hidden_state is None:
#             h_t = torch.zeros(batch_size, self.hidden_size, device=input_sequence.device)
#         else:
#             h_t = initial_hidden_state
#         if initial_cell_state is None:
#             c_t = torch.zeros(batch_size, self.hidden_size, device=input_sequence.device)
#         else:
#             c_t = initial_cell_state

#         outputs = []
#         for t in range(seq_len):
#             # Input at current step could be a combination of actual input and previous output
#             current_input = input_sequence[:, t, :] # Example: using original input
#             # If previous output is to be fed back, modify current_input here
#             # e.g., current_input = torch.cat((input_sequence[:, t, :], self.linear(h_t)), dim=1) 
#             # (requires adjusting input_size of LSTMCell)

#             h_t, c_t = self.lstm_cetorch.stack(outputs, dim=1)ll(current_input, (h_t, c_t))
#             output = self.linear(h_t)
#             outputs.append(output)

#         return 

# # Example usage:
# input_size = 10
# hidden_size = 20
# output_size = 5
# batch_size = 2
# seq_len = 7

# model = CustomLSTMModel(input_size, hidden_size, output_size)
# dummy_input = torch.randn(batch_size, seq_len, input_size)
# output = model(dummy_input)
# print(output.shape) # Expected: (batch_size, seq_len, output_size)



# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # LSTM layer
#         self.lstm = nn.LSTM(input_size=input_size,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True)
        
#         # Output layer
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         """
#         x: input of shape (batch_size, seq_length, input_size)
#         """
#         # LSTM returns (output, (h_n, c_n))
#         lstm_out, (h_n, c_n) = self.lstm(x)
        
#         # Take output from last time step
#         last_output = lstm_out[:, -1, :]
        
#         # Pass through output layer
#         output = self.fc(last_output)
        
#         return output
# # Create LSTM model
# lstm_model = LSTMModel(input_size=10, hidden_size=20, output_size=5, num_layers=2)
# print("LSTM Model created")
# # Test with sample input
# x = torch.randn(4, 8, 10)  # (batch_size=4, seq_length=8, input_size=10)
# output = lstm_model(x)
# print("LSTM Output shape:", output.shape)
# # Output: LSTM Output shape: torch.Size([4, 5])
# # Compare outputs (LSTM should produce more stable gradients)
# print("LSTM output sample:", output[0])
# # Output: LSTM output sample: tensor([-0.1234, 0.0567, -0.0891, 0.1123, -0.0456], grad_fn=<AddBackward0>)




