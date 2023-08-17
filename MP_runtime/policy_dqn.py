
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

  def __init__(self):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(4, 64)
    self.fc3 = nn.Linear(64, 2)

  def forward(self, x):
    x = F.elu(self.fc1(x))
    x = self.fc3(x)
    return x


