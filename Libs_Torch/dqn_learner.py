import gym
import numpy as np
import argparse
from itertools import count
from collections import deque
import time
# import Config

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

seed = 0
test_episodes = 100
test_every = 50000
number_of_steps = 1000000
buffer_size = 50000
min_buffer_size = 1000
batch_size = 64

start_steps = 10000

hidden_sizes = [256, 256]
gamma = 0.99

decay = True
policy_lr = 0.0002
critic_lr = 0.0003

adam_eps = 1e-7
# Used to update target networks
polyak = 0.995
env_scale = True

writer_flag = False

class DQNN(nn.Module):

  def __init__(self, input_state, output_action):
    super(DQNN, self).__init__()
    # self.fc1 = nn.Linear(input_state, Config.hidden_sizes[0])
    # self.fc3 = nn.Linear(Config.hidden_sizes[0], output_action)
    self.fc1 = nn.Linear(input_state, hidden_sizes[0])
    self.fc3 = nn.Linear(hidden_sizes[0], output_action)

  def forward(self, x):
    x = F.elu(self.fc1(x))
    x = self.fc3(x)
    return x
    
  def get_parameters_as_lists(self):
    # Extract and convert weights and biases to lists for sycl exe
    parameters_as_lists = [param.data.tolist() for param in self.parameters()]
    return parameters_as_lists

class DQNLearner:
    def __init__(self, input_state, output_action, device='cpu'):
        self.device = device
        # self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.action_shape = output_action
        self.policy_nn = DQNN(input_state, output_action).to(self.device)
        self.target_nn = DQNN(input_state, output_action).to(self.device)
        self.target_nn.load_state_dict(self.policy_nn.state_dict())

        self.mse = torch.nn.MSELoss()
        # self.optimizer = optim.RMSprop(policy_nn.parameters(), lr=Config.policy_lr, weight_decay=1e-4)
        self.optimizer = optim.RMSprop(self.policy_nn.parameters(), lr=policy_lr, weight_decay=1e-4)

        self.noise_std = 0.1

    def get_action(self, state, epsilon):
        with torch.no_grad():
            policy_net_cpu=copy.deepcopy(self.policy_nn).to('cpu')
            greedy_action = torch.argmax(policy_net_cpu(state)).item()
            random_action = np.random.randint(0, 2)
        return random_action if np.random.rand() < epsilon else greedy_action

    def update_policy(self, states, actions, next_states, rewards, dones):
        t1=time.perf_counter()
        state_action_values0 = self.policy_nn(states.to(self.device))
        state_action_values1 = state_action_values0.gather(1, actions[:, None].long())
        state_action_values = state_action_values1.squeeze().to(self.device)
        next_state_values = torch.max(self.target_nn(next_states.to(self.device)), dim=1)[0].detach()
        expected_state_action_values = (rewards + next_state_values * (~dones) * gamma).to(self.device)

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        t2=time.perf_counter()
        return t2-t1, loss

    def update_targets(self):
        self.target_nn.load_state_dict(self.policy_nn.state_dict())


        #inputs (states, actions, next_states, rewards, dones): from Replay Buffer
    def update_all_gradients(self, states, actions, next_states, rewards, dones, bool_targ_upd):
        t, l = self.update_policy(states, actions, next_states, rewards, dones)
        if (bool_targ_upd):
            self.update_targets()
        return t, l, self.policy_nn.state_dict()

    def update_network_multistream(self, states, actions, next_states, rewards, dones):
        t1 = time.perf_counter()
        
        batch_size = states.shape[0]
        C = 2  # Number of CUDA streams
        
        streams = [torch.cuda.Stream() for _ in range(C)]
        chunk_size = (batch_size + C - 1) // C
        
        state_action_values_list = []
        next_state_values_list = []
        expected_state_action_values_list = []
        loss_list = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            with torch.cuda.stream(streams[i % C]):
                chunk_states = states[i:end_idx]
                chunk_actions = actions[i:end_idx]
                chunk_next_states = next_states[i:end_idx]
                chunk_rewards = rewards[i:end_idx]
                chunk_dones = dones[i:end_idx]

                state_action_values = self.policy_nn(chunk_states).gather(1, chunk_actions[:, None].long()).squeeze()
                next_state_values = torch.max(self.target_nn(chunk_next_states), dim=1)[0].detach()
                expected_state_action_values = chunk_rewards + next_state_values * (1 - chunk_dones) * cfg.gamma

                loss = F.mse_loss(state_action_values, expected_state_action_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                state_action_values_list.append(state_action_values)
                next_state_values_list.append(next_state_values)
                expected_state_action_values_list.append(expected_state_action_values)
                loss_list.append(loss)

        torch.cuda.synchronize()
        # print("here")
        state_action_values = torch.cat(state_action_values_list)
        next_state_values = torch.cat(next_state_values_list)
        expected_state_action_values = torch.cat(expected_state_action_values_list)
        loss = torch.stack(loss_list).mean()
        
        t2 = time.perf_counter()
        
        return t2 - t1, loss

# policy_in_dim=4
# policy_out_dim=2
# train_batch_size=128
# DLearner = Learner(policy_in_dim, policy_out_dim)
# states = torch.rand(train_batch_size, policy_in_dim)
# actions = torch.rand(train_batch_size)
# next_states = torch.rand(train_batch_size, policy_in_dim)
# rewards = torch.rand(train_batch_size)
# dones = torch.zeros(train_batch_size)
# start = time.perf_counter()
# DLearner.update_all_gradients(1, states, actions, next_states, rewards, dones, False)
# print("yes")