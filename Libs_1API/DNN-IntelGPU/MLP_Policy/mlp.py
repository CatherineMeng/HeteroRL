#!/usr/bin/env python
# encoding: utf-8

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