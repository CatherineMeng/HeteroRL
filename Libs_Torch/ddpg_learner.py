import gym
# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import Config
import torch
from torch import nn
import numpy as np
import time


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


class DDPGLearner:
    def __init__(self, input_state, output_action, device='cuda'):
        self.device = device
        # self.device ='cuda' if torch.cuda.is_available() else 'cpu'
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
        target = rewards + Config.gamma * target_values * (~dones)
        bs = actions.size(dim=0)
        state_values = self.moving_critic_nn(states, actions.view(bs,1)).squeeze(-1)
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
                targ.data.add_((1-Config.polyak) * mov.data)

            for mov, targ in zip(self.moving_policy_nn.parameters(), self.target_policy_nn.parameters()):
                targ.data.mul_(Config.polyak)
                targ.data.add_((1-Config.polyak) * mov.data)

    #inputs (states, actions, next_states, rewards, dones): from Replay Buffer
    def update_all_gradients(self, states, actions, next_states, rewards, dones, bool_targ_upd):
        t1=time.perf_counter()
        # Implement learning rate decay for both NNs and std decay for Random function
        self.lr_std_decay(30)

        # Calculate loss and update moving critic
        critic_loss = self.update_critic(states, actions,rewards, next_states, dones)
        # Calculate loss and update moving policy
        policy_loss = self.update_policy(states)
        # Update target policy and critic to slowly follow moving NNs with polyak averaging
        if (bool_targ_upd):
            self.update_targets()
        # self.critic_loss_mean.append(critic_loss)
        # self.policy_loss_mean.append(policy_loss)
        t2=time.perf_counter()
        return t2-t1,policy_loss,self.moving_policy_nn.state_dict()