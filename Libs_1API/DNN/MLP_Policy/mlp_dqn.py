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

# from drawnow import drawnow
import matplotlib.pyplot as plt

# from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU if available
    print("USING GPU")

# device = torch.device("cpu")
# torch.set_num_threads(1)



parser = argparse.ArgumentParser(description='PyTorch DQN solution of CartPole-v0')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--epsilon_start', type=float, default=0.9)
parser.add_argument('--epsilon_end', type=float, default=0.05)
parser.add_argument('--target_update', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--max_episode', type=int, default=10)

cfg = parser.parse_args()

last_score_plot = [0]*cfg.max_episode
avg_score_plot = [0]*cfg.max_episode


env = gym.make('CartPole-v0')


class Memory(object):
  def __init__(self, memory_size=10000):
    self.memory = deque(maxlen=memory_size)
    self.memory_size = memory_size

  def __len__(self):
    return len(self.memory)

  def append(self, item):
    self.memory.append(item)

  def sample_batch(self, batch_size):
    idx = np.random.permutation(len(self.memory))[:batch_size]
    return [self.memory[i] for i in idx]


class DQN(nn.Module):

  def __init__(self,input_channels, hidden_size, num_actions):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(input_channels, hidden_size)
    self.fc3 = nn.Linear(hidden_size, num_actions)

  def forward(self, x):
    x = F.elu(self.fc1(x))
    x = self.fc3(x)
    return x

class Trainer:
    def __init__(self, input_channels, hidden_size, num_actions):
        self.policy_net = DQN(input_channels,hidden_size, num_actions)
        self.policy_net.to(device)
        self.target_net = DQN(input_channels, hidden_size,num_actions)
        self.target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.num_actions=num_actions
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr, weight_decay=1e-4)

    def get_action(self, state, epsilon):
      with torch.no_grad():
        # greedy_action = torch.argmax(policy_net(state), dim=1).item()
        policy_net_cpu=copy.deepcopy(self.policy_net).to('cpu')
        greedy_action = torch.argmax(policy_net_cpu(state)).item()
        random_action = np.random.randint(0, 2)
      return random_action if np.random.rand() < epsilon else greedy_action

    def sync_target(self):
      self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_network(self, states, actions, next_states, rewards, dones):
      t1=time.perf_counter()
      state_action_values = self.policy_net(states).gather(1, actions[:, None].long()).squeeze()
      next_state_values = torch.max(self.target_net(next_states), dim=1)[0].detach()
      expected_state_action_values = rewards + next_state_values * (1 - dones) * cfg.gamma

      loss = F.mse_loss(state_action_values, expected_state_action_values)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
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

                state_action_values = self.policy_net(chunk_states).gather(1, chunk_actions[:, None].long()).squeeze()
                next_state_values = torch.max(self.target_net(chunk_next_states), dim=1)[0].detach()
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


# Random-data Testbench
if __name__ == "__main__":

  memory = Memory(10000)
  trainer = Trainer(4,64,2)
  train_time_epsavg_total=0
  # for i in tqdm(range(cfg.max_episode)):
  t_start=time.perf_counter()
  for i in range(cfg.max_episode):
    episode_durations = 0
    epsilon = (cfg.epsilon_end - cfg.epsilon_start) * (i / cfg.max_episode) + cfg.epsilon_start

    train_time_total=0
    for t in count():

      # Random data for testing. Replace with env of DRL config
      state = np.random.rand(4)
      action = trainer.get_action(torch.tensor(state).float(), epsilon)
      next_state = np.random.rand(4)
      reward = 1.0
      done = False

      memory.append([state, action, next_state, reward, done])
      # state = next_state

      if len(memory) > cfg.batch_size:
        states, actions, next_states, rewards, dones = \
          map(lambda x: torch.tensor(x).float(), zip(*memory.sample_batch(cfg.batch_size)))

        p_time=trainer.update_network(states.to(device), actions.to(device), next_states.to(device), rewards.to(device), dones.to(device))
        # multistream tested, :D
        # p_time=update_network_multistream(states.to(device), actions.to(device), next_states.to(device), rewards.to(device), dones.to(device))

        train_time_total+=p_time

      if done or t==200:
        episode_durations = t + 1
        avg_score_plot[i]=(avg_score_plot[i-1] * 0.99 + episode_durations * 0.01)
        last_score_plot[i]=(episode_durations)
        train_time_epsavg_total+=1000*train_time_total/episode_durations #in ms
      #   print("Episode",i,"- Avg gradient update time of batch",cfg.batch_size,"is",1000*train_time_total/episode_durations,"ms")
        break

    # Update the target network
    if i % cfg.target_update == 0:
      trainer.sync_target()


  print("total time for", cfg.max_episode,"training episodes:",time.perf_counter()-t_start,"seconds")
  print("train_time_epsavg_total:",train_time_epsavg_total)
  print("Batch",cfg.batch_size," - avg per-gradeint-update train time:",train_time_epsavg_total/cfg.max_episode)


  print('Complete')
  env.close()
