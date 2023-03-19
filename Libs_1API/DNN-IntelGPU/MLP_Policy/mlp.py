#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2019 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import intel_extension_for_pytorch as ipex


# BS_TRAIN: Batch size for training data
# BS_TEST:  Batch size for testing data
# EPOCHNUM: Number of epoch for training

BS_TRAIN = 50
BS_TEST  = 10
EPOCHNUM = 2


# TestDataset class is inherited from torch.utils.data.Dataset.
# Since data for training involves data and ground truth, a flag "train" is defined in the initialization function. When train is True, instance of TestDataset gets a pair of training data and label data. When it is False, the instance gets data only for inference. Value of the flag "train" is set in __init__ function.
# In __getitem__ function, data at index position is supposed to be returned.
# __len__ function returns the overall length of the dataset.

class TestDataset(Dataset):
    def __init__(self, train = True):
        super(TestDataset, self).__init__()
        self.train = train

    def __getitem__(self, index):
        if self.train:
            return torch.rand(3, 112, 112), torch.rand(6, 110, 110)
        else:
            return torch.rand(3, 112, 112)

    def __len__(self):
        if self.train:
            return 100
        else:
            return 20


# TestModel class is inherited from torch.nn.Module.
# Operations that will be used in the topology are defined in __init__ function.
# Input data x is supposed to be passed to the forward function. The topology is implemented in the forward function. When perform training/inference, the forward function will be called automatically by passing input data to a model instance.

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(3, 6, 3)
        self.norm = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


# Perform training and inference in main function

def main():
    
    # The following 3 components are required to perform training.
    # 1. model: Instantiate model class
    # 2. optim: Optimization function for update topology parameters during training
    # 3. crite: Criterion function to minimize loss
    
    model = TestModel()
    model = model.to("xpu", memory_format=torch.channels_last)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    crite = nn.MSELoss(reduction='sum')


    # 1. Instantiate the Dataset class defined before
    # 2. Use torch.utils.data.DataLoader to load data from the Dataset instance
    
    train_data  = TestDataset()
    trainLoader = DataLoader(train_data, batch_size=BS_TRAIN)
    test_data   = TestDataset(train=False)
    testLoader  = DataLoader(test_data, batch_size=BS_TEST)


    # Apply Intel Extension for PyTorch optimization against the model object and optimizer object.

    model, optim = ipex.optimize(model, optimizer=optim)


    # Perform training and inference
    # Use model.train() to set the model into train mode. Use model.eval() to set the model into inference mode.
    # Use for loop with enumerate(instance of DataLoader) to go through the whole dataset for training/inference.

    for i in range(0, EPOCHNUM - 1):

        # Iterate dataset for training to train the model

        model.train()
        for batch_index, (data, y_ans) in enumerate(trainLoader):
            data = data.to("xpu", memory_format=torch.channels_last)
            y_ans = y_ans.to("xpu", memory_format=torch.channels_last)
            optim.zero_grad()
            y = model(data)
            loss = crite(y, y_ans)
            loss.backward()
            optim.step()

   
        # Iterate dataset for validation to evaluate the model
 
        model.eval()
        for batch_index, data in enumerate(testLoader):
            data = data.to("xpu", memory_format=torch.channels_last)
            y = model(data)

if __name__ == '__main__':
    main()
    print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')
'''


import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as optim

import copy
import torch
from torch.autograd import Variable
import random

state_dim = 4
action_dim = 2

class DQN(nn.Module):
    def __init__(self, state_dim, num_actions, lr=0.05):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(state_dim, 32)
        self.output = nn.Linear(32, num_actions)
        # each action is associated with a unique q value
        # for cartpole, state_dim=4, num_actions=2
        self.criterion = torch.nn.MSELoss()
        

    def forward(self, input):
        input = F.relu(self.linear1(input))
        input = self.output(input)
        return input

    def act(self, state):
        q_value = self.forward(state)
        action = q_value.max(1)[1]
        return action
    
def update(model, dqn_optimizer, state, y):
    """Update the weights of the network given a batch of expereinces, y is the batch of q values. """
    # state = state.to("xpu")
    # y = y.to("xpu")

    y_pred = model.forward(torch.Tensor(state))
    loss = model.criterion(y_pred, Variable(torch.Tensor(y)))
    # self.optimizer.zero_grad()
    dqn_optimizer.zero_grad()
    loss.backward()
    dqn_optimizer.step()

# def xavier_init(module):
#     with torch.no_grad():
#         if isinstance(module, nn.Linear):
#             init.xavier_normal_(module.weight)
#             init.constant_(module.bias, 0.01)

def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=20, 
               title = 'DQL', double=False, 
               n_update=10, soft=False, verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    dqn_optimizer = optim.Adam(model.parameters(), lr)
    final = []
    memory = []
    episode_i=0
    sum_total_replay_time=0
    for episode in range(episodes):
        episode_i+=1
        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()
        if double and soft:
            model.target_update()
        
        # Reset state
        # state = env.reset()
        state = torch.rand((1, state_dim))
        done = False
        total = 0
        
        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                # action = env.action_space.sample()
                action = random.randint(0, 1)
            else:
                q_values = model.forward(state)
                action = torch.argmax(q_values).item()
            
            # Take action and add reward to total
            # next_state, reward, done, _ = env.step(action)
            next_state = torch.rand((1, state_dim))
            reward = random.randint(0, 1)
            done= random.randint(0, 1)

            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))
            q_values = model.forward(state).tolist()
            print("q_values: ",q_values)
             
            if done:
                if not replay:
                    q_values[0][action] = reward
                    # Update network weights
                    update(model, dqn_optimizer, state, q_values)
                break

            if replay:
                t0=time.time()
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma)
                t1=time.time()
                sum_total_replay_time+=(t1-t0)
            else: 
                # Update network weights using the last step only
                q_values_next = model.forward(next_state)
                q_values[0][action] = reward + gamma * torch.max(q_values_next).item()
                update(model, dqn_optimizer,state, q_values)

            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        # plot_res(final, title)
        
        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
            if replay:
                print("Average replay time:", sum_total_replay_time/episode_i)
        
    return final

if __name__ == "__main__":

    # Number of episodes
    episodes = 10
    # Learning rate
    lr = 0.001
    # env not relevant here, interface with gym later
    env="cp"

    # Get DQN results
    simple_dqn = DQN(state_dim, action_dim, lr)
    # simple_dqn = simple_dqn.to("xpu")
    simple = q_learning(env, simple_dqn, episodes, gamma=.9, epsilon=0.3)

    print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')