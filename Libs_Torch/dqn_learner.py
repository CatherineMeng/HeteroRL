import gym
import numpy as np
import argparse
from itertools import count
from collections import deque
import time

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PolicyNN(nn.Module):

  def __init__(self, input_state, output_action):
    super(PolicyNN, self).__init__()
    self.fc1 = nn.Linear(input_state, Config.hidden_sizes[0])
    self.fc3 = nn.Linear(Config.hidden_sizes[0], output_action)

  def forward(self, x):
    x = F.elu(self.fc1(x))
    x = self.fc3(x)
    return x

class Learner:
    def __init__(self, input_state, output_action, device='cuda'):
        self.device = device
        # self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.action_shape = output_action
        self.policy_nn = PolicyNN(input_state, output_action).to(self.device)
        self.target_nn = PolicyNN(input_state, output_action).to(self.device)
        self.target_nn.load_state_dict(self.policy_nn.state_dict())

        self.mse = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(policy_nn.parameters(), lr=Config.policy_lr, weight_decay=1e-4)

        self.noise_std = 0.1

    def get_action(self, state, epsilon):
        with torch.no_grad():
            policy_net_cpu=copy.deepcopy(self.policy_nn).to('cpu')
            greedy_action = torch.argmax(policy_net_cpu(state)).item()
            random_action = np.random.randint(0, 2)
        return random_action if np.random.rand() < epsilon else greedy_action

    def update_policy(self, states, actions, next_states, rewards, dones):
        t1=time.perf_counter()
        state_action_values = self.policy_nn(states).gather(1, actions[:, None].long()).squeeze()
        next_state_values = torch.max(self.target_nn(next_states), dim=1)[0].detach()
        expected_state_action_values = rewards + next_state_values * (1 - dones) * cfg.gamma

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2=time.perf_counter()
        return t2-t1

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
    
    return t2 - t1


    def update_targets(self):
        self.target_nn.load_state_dict(self.policy_nn.state_dict())

    def lr_std_decay(self, n_step):
        if Config.decay:
            frac = 1 - n_step / Config.number_of_steps
            self.policy_nn_optim.param_groups[0]["lr"] = frac * Config.policy_lr
            self.critic_nn_optim.param_groups[0]["lr"] = frac * Config.critic_lr
            self.noise_std = self.noise_std * frac

    #inputs (states, actions, next_states, rewards, dones): from Replay Buffer
    def update_all_gradients(self, n_step, states, actions, next_states, rewards, dones, bool_targ_upd):
        # Implement learning rate decay for both NNs and std decay for Random function
        self.lr_std_decay(n_step)
        self.update_policy(self, states, actions, next_states, rewards, dones)
        if (bool_targ_upd):
            self.update_targets()
