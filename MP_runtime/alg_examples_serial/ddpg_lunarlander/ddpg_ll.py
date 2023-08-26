import gym
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import Config
import torch
from torch import nn
import numpy as np

import itertools

from collections import deque


class Buffer:
    def __init__(self, state_size, action_size, buffer_capacity=100000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.states = torch.zeros(buffer_capacity, state_size).to(self.device)
        self.actions = torch.zeros(buffer_capacity, action_size).to(self.device)
        self.new_states = torch.zeros(buffer_capacity, state_size).to(self.device)
        self.rewards = torch.zeros(buffer_capacity).to(self.device)
        self.dones = torch.zeros(buffer_capacity).to(self.device)
        self.buffer_counter = 0
        self.initialized = False
        self.buffer_size = buffer_capacity

    def add(self, state, actions, new_state, reward, done):

        self.states[self.buffer_counter] = torch.Tensor(state).to(self.device)
        self.actions[self.buffer_counter] = torch.Tensor(actions).to(self.device)
        self.new_states[self.buffer_counter] = torch.Tensor(new_state).to(self.device)
        self.rewards[self.buffer_counter] = torch.Tensor((reward,)).squeeze(-1).to(self.device)
        self.dones[self.buffer_counter] = torch.Tensor((1 if done else 0,)).squeeze(-1).to(self.device)

        self.buffer_counter = (self.buffer_counter + 1) % self.buffer_size
        if self.buffer_counter == 0 and not self.initialized:
            self.initialized = True

    def sample_indices(self, batch_size):
        indices = np.arange(min(self.buffer_counter, self.buffer_size) if not self.initialized else self.buffer_size)
        np.random.shuffle(indices)
        indices = indices[:batch_size]
        return indices

torch.manual_seed(Config.seed)
class PolicyNN(nn.Module):
    def __init__(self, input_state, output_action):
        super(PolicyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_state, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[0], Config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], output_action),
            nn.Tanh()
        )

    def forward(self, state):
        actions = self.model(state)
        return actions

class CriticNN(nn.Module):
    def __init__(self, input_state, input_actions):
        super(CriticNN, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(input_state + input_actions, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[0], Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[0], 1)
        )

    def forward(self, state, actions):
        input_state_action = torch.cat((state, actions), 1)
        return self.value(input_state_action)




class AgentControl:
    def __init__(self, input_state, output_action):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_shape = output_action
        self.moving_policy_nn = PolicyNN(input_state, output_action).to(self.device)
        self.policy_nn_optim = torch.optim.Adam(params=self.moving_policy_nn.parameters(), lr=Config.policy_lr,
                                                eps=Config.adam_eps)
        self.moving_critic_nn = CriticNN(input_state, output_action).to(self.device)
        self.critic_nn_optim = torch.optim.Adam(params=self.moving_critic_nn.parameters(), lr=Config.critic_lr,
                                                eps=Config.adam_eps)
        self.target_policy_nn = PolicyNN(input_state, output_action).to(self.device)
        self.target_policy_nn.load_state_dict(self.moving_policy_nn.state_dict())
        self.target_critic_nn = CriticNN(input_state, output_action).to(self.device)
        self.target_critic_nn.load_state_dict(self.moving_critic_nn.state_dict())
        self.mse = torch.nn.MSELoss()

        self.noise_std = 0.1

    def get_action(self, state):
        actions = self.moving_policy_nn(torch.Tensor(state).to(self.device))
        noise = (self.noise_std ** 0.5) * torch.randn(self.action_shape).to(self.device)
        return np.clip((actions + noise).cpu().detach().numpy(), -1, 1)

    def lr_std_decay(self, n_step):
        if Config.decay:
            frac = 1 - n_step / Config.number_of_steps
            self.policy_nn_optim.param_groups[0]["lr"] = frac * Config.policy_lr
            self.critic_nn_optim.param_groups[0]["lr"] = frac * Config.critic_lr
            self.noise_std = self.noise_std * frac

    def update_critic(self, states, actions, rewards, new_states, dones):
        # Using target policy NN find actions for new states, concatenate them to new state Tensor and feed it to
        # target critic NN to get Q values from new states and new actions
        new_actions = self.target_policy_nn(new_states).detach()
        target_values = self.target_critic_nn(new_states, new_actions).squeeze(-1).detach()
        # Calculate target with reward and estimated value from Q function of new state. If its the end of the episode
        # calculate target only with reward. Also erase gradients from target tensors because we want to update
        # only moving critic NN.
        target = rewards + Config.gamma * target_values * (1 - dones)
        state_values = self.moving_critic_nn(states, actions).squeeze(-1)
        critic_loss = self.mse(state_values, target)

        self.critic_nn_optim.zero_grad()
        critic_loss.backward()
        self.critic_nn_optim.step()

        return critic_loss.cpu().detach().numpy()

    def update_policy(self, states):
        policy_actions = self.moving_policy_nn(states)
        critic_value = self.moving_critic_nn(states, policy_actions).squeeze(-1)
        # Used `-value` as we want to maximize the value given by the critic for our actions
        policy_loss = -torch.mean(critic_value)

        self.policy_nn_optim.zero_grad()
        policy_loss.backward()
        self.policy_nn_optim.step()

        return policy_loss.cpu().detach().numpy()

    def update_targets(self):
        # Update target networks by polyak averaging.
        with torch.no_grad():
            for mov, targ in zip(self.moving_critic_nn.parameters(), self.target_critic_nn.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                targ.data.mul_(Config.polyak)
                targ.data.add_((1 - Config.polyak) * mov.data)

            for mov, targ in zip(self.moving_policy_nn.parameters(), self.target_policy_nn.parameters()):
                targ.data.mul_(Config.polyak)
                targ.data.add_((1 - Config.polyak) * mov.data)

class Agent:
    # Role of Agent class is to coordinate between AgentControll where we do all calculations
    # and Buffer where we store all of the data
    def __init__(self, state_size, action_size):
        self.agent_control = AgentControl(state_size, action_size)
        self.buffer = Buffer(state_size, action_size, buffer_capacity=Config.buffer_size)
        self.critic_loss_mean = deque(maxlen=100)
        self.policy_loss_mean = deque(maxlen=100)
        self.max_reward = -300
        self.ep_count = -1
        self.ep_tested = -1

    def get_action(self, state, n_step, env):
        # For better exploration, agent will take random actions for first Config.start_steps number of steps
        if n_step < Config.start_steps:
            return env.action_space.sample()
        else:
            return self.agent_control.get_action(state)

    def add_to_buffer(self, state, actions, new_state, reward, done):
        self.buffer.add(state, actions, new_state, reward, done)

    def update(self, n_step):
        # Implement learning rate decay for both NNs and std decay for Random function
        self.agent_control.lr_std_decay(n_step)
        # Wait until buffer has enough of data
        if self.buffer.buffer_counter < Config.min_buffer_size and not self.buffer.initialized:
            return
        # Get indices of randomly selected steps
        indices = self.buffer.sample_indices(Config.batch_size)
        # Calculate loss and update moving critic
        critic_loss = self.agent_control.update_critic(self.buffer.states[indices], self.buffer.actions[indices],
                                                       self.buffer.rewards[indices], self.buffer.new_states[indices],
                                                       self.buffer.dones[indices])
        # Calculate loss and update moving policy
        policy_loss = self.agent_control.update_policy(self.buffer.states[indices])
        # Update target policy and critic to slowly follow moving NNs with polyak averaging
        self.agent_control.update_targets()
        self.critic_loss_mean.append(critic_loss)
        self.policy_loss_mean.append(policy_loss)

    def record_results(self, n_step, writer, env):
        if self.buffer.buffer_counter < Config.min_buffer_size and not self.buffer.initialized or self.ep_count == env.episode_count:
            return
        self.ep_count = env.episode_count
        self.max_reward = np.maximum(self.max_reward, np.max(env.return_queue))
        print("Ep " + str(self.ep_count) + " St " + str(n_step) + "/" + str(Config.number_of_steps) + " Mean 100 policy loss: " + str(
            np.round(np.mean(self.policy_loss_mean), 4)) + " Mean 100 critic loss: " + str(
            np.round(np.mean(self.critic_loss_mean), 4)) + " Max reward: " + str(
            np.round(self.max_reward, 2)) + " Mean 100 reward: " + str(
            np.round(np.mean(env.return_queue), 2)) + " Last rewards: " + str(
            np.round(env.return_queue[-1], 2)))

        if Config.writer_flag:
            writer.add_scalar('pg_loss', np.mean(self.policy_loss_mean), self.ep_count)
            writer.add_scalar('vl_loss', np.mean(self.critic_loss_mean), self.ep_count)
            writer.add_scalar('100rew', np.mean(env.return_queue), self.ep_count)
            writer.add_scalar('rew', env.return_queue[-1], self.ep_count)



# --------------------------------------------------- Initialization ---------------------------------------------------
# Create Lunar Lander enviroment and add wrappers to record statistics
env = gym.make(Config.env_name)
env = gym.wrappers.RecordEpisodeStatistics(env)
state = env.reset()
state=state[0]
# Create agent which will use DDPG to train NNs
agent = Agent(state.shape[0], env.action_space.shape[0])

# Create writer for Tensorboard
writer = SummaryWriter(log_dir='content/runs/'+Config.writer_name) if Config.writer_flag else None
print(Config.writer_name)
# ------------------------------------------------------ Training ------------------------------------------------------
for n_step in range(Config.number_of_steps):

    #env.render()
    # Feed current state to the policy NN and get action
    actions = agent.get_action(state, n_step, env)
    # Use given action and retrieve new state, reward agent recieved and whether episode is finished flag
    new_state, reward, done, _,_ = env.step(actions)
    # Store step information to buffer for future use
    agent.add_to_buffer(state, actions, new_state, reward, done)
    # Update all 4 NNs
    agent.update(n_step)
    state = new_state
    if done:
        state = env.reset()
        state=state[0]
    # Print results to console and Tensorboard Writer
    agent.record_results(n_step, writer, env)
if writer is not None:
    writer.close()
test_process.env.close()
env.close()

#tensorboard --logdir="D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\drl-ddpg-lunar-lander\content\runs" --host=127.0.0.1