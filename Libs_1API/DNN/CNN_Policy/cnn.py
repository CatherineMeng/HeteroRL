import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import time

device = torch.device("cpu")
torch.set_num_threads(16)
# if torch.cuda.is_available():
#     device = torch.device("cuda")  # Use GPU if available
#     print("USING GPU")
#     torch.set_num_threads(1)

    
# device = torch.device("cpu")
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        # Input one state: {1, 3, 210, 160}
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(64 * 22 * 16, 512)
        self.output = nn.Linear(512, num_actions)

    def forward(self, input):
        input = torch.relu(self.conv1(input))
        input = torch.relu(self.conv2(input))
        input = torch.relu(self.conv3(input))
        # Flatten the output
        input = input.view(input.size(0), -1)
        input = torch.relu(self.linear1(input))
        input = self.output(input)
        return input

    def act(self, state):
        q_value = self.forward(state)
        action = q_value.argmax(dim=1)
        return action

class Trainer:
    def __init__(self, input_channels, num_actions, capacity):
        self.network = DQN(input_channels, num_actions)
        self.network.to(device)
        self.target_network = DQN(input_channels, num_actions)
        self.target_network.to(device)
        self.dqn_optimizer = optim.Adam(self.network.parameters(), lr=0.0001)
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 30000
        self.batch_size = 512
        self.gamma = 0.99
        self.num_actions=num_actions
        # self.buffer = ExperienceReplay(capacity)  # Uncomment and implement the ExperienceReplay class.

    def compute_td_loss(self, batch_size, gamma):
        # Code for buffer sampling is commented out as it's not implemented here.
        # Replace it with actual buffer implementation later if needed.

        # (Assuming batch is obtained from the buffer as commented out in the code)
        states_tensor = torch.rand((batch_size, 3, 210, 160), requires_grad=True).to(device)
        new_states_tensor = torch.rand((batch_size, 3, 210, 160), requires_grad=True).to(device)
        actions_tensor = torch.rand((batch_size), requires_grad=True).to(device)
        rewards_tensor = torch.rand((batch_size), requires_grad=True).to(device)
        dones_tensor = torch.rand((batch_size), requires_grad=True).to(device)

        # print("states_tensor.shape:",states_tensor.shape)
        # print("new_states_tensor.shape:",new_states_tensor.shape)
        # print("actions_tensor.shape:",actions_tensor.shape)
        # print("rewards_tensor.shape:",rewards_tensor.shape)
        # print("dones_tensor.shape:",dones_tensor.shape)

        t1=time.perf_counter()
        q_values = self.network.forward(states_tensor)
        next_target_q_values = self.target_network.forward(new_states_tensor)
        next_q_values = self.network.forward(new_states_tensor)

        actions_tensor = actions_tensor.to(torch.long)

        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        _, maximum = next_q_values.max(1)
        next_q_value = next_target_q_values.gather(1, maximum.unsqueeze(1)).squeeze(1)
        expected_q_value = rewards_tensor + gamma * next_q_value * (1 - dones_tensor)

        loss = nn.MSELoss()(q_value, expected_q_value)

        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        t2=time.perf_counter()

        return t2-t1, loss.item()

    def epsilon_by_frame(self, frame_id):
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1.0 * frame_id / self.epsilon_decay)
        return epsilon
        
    # The rest of the Trainer class methods are not implemented here for simplicity.

    def train(self, random_seed, rom_path, num_epochs, num_actions):

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Load the environment if required.
        # load_enviroment(random_seed, rom_path)

        # The following code is commented out since it's not clear how the environment is used in the original code.
        # If needed, modify it to interact with the environment according to the available actions.
        legal_actions = list(range(num_actions))

        episode_reward = 0.0
        all_rewards = []
        # losses = []
        start = time.time()

        sum_time=0
        for epoch in range(1, num_epochs + 1):
            epsilon = self.epsilon_by_frame(epoch)
            if random.random() <= epsilon:
                action = random.choice(legal_actions)
            else:
                state_tensor = torch.rand((1, 3, 210, 160)).to(device)  # Replace with actual environment observation later.

                action_tensor = self.network.act(state_tensor)
                action = action_tensor.item()

            # For demonstration purposes, reward and done are randomly generated.
            reward = random.random()  # Replace with actual environment reward later.
            episode_reward += reward
            done = bool(random.getrandbits(1))  # Replace with actual environment "done" status later.

            next_state_tensor = torch.rand((1, 3, 210, 160)).to(device)  # Replace with actual environment observation later.

            reward_tensor = torch.tensor(reward).to(device)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(device)
            action_tensor_new = torch.tensor(action, dtype=torch.long).to(device)

            # If using the experience replay buffer, uncomment the following line to push the experience.
            # buffer.push(state_tensor, next_state_tensor, action_tensor_new, done_tensor, reward_tensor)

            state_tensor = next_state_tensor

            if done:
                state_tensor = torch.rand((1, 3, 210, 160)).to(device)  # Renew state if done.
                # all_rewards.append(episode_reward)
                episode_reward = 0.0

            # if epoch >= 8192: #start after filling replay. not necessary here.
                # loss = self.compute_td_loss(self.batch_size, self.gamma)
                # losses.append(loss)
            ptime, loss = self.compute_td_loss(self.batch_size, self.gamma)
            if (epoch!=1):
                sum_time +=ptime

            print("per-gradient-step with batch",self.batch_size,":",ptime*1000,"ms")

            if epoch % 10 == 0:
                # print(episode_reward)
                self.target_network.load_state_dict(self.network.state_dict())

        print("avg time with batch",self.batch_size,":",sum_time*1000/(num_epochs-1),"ms")
        stop = time.time()
        print("Time taken by function:", stop - start, "seconds")


if __name__ == "__main__":
    input_channels = 3
    num_actions = 18
    capacity = 8192
    trainer = Trainer(input_channels, num_actions, capacity)
    trainer.train(123, "/Users/b.bin", 10, num_actions)