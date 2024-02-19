
from sycl_rm_module import SumTreeNary
import time 
import random
import torch

class replay_top():
    def __init__(self, fanout, train_bs, insert_bs, memory_size=10000):
        self.RM = SumTreeNary(memory_size, fanout) 
        self.memory = {key: None for key in range(memory_size)} #data storage
        self.memory_size = memory_size
        self.ibs = insert_bs
        self.tbs = train_bs
        self.memoize_indices = [None]*train_bs
        

    def __len__(self):
        return len(self.memory)

    # in both sum tree and data storage
    # indices, items and new_prs have size insert_bs
    # each item will follow the format [state, action, next_state, reward, done]
    def insert_through(self, items, new_prs):
        indices = [None] * self.ibs
        for i,item in enumerate(items):
            self.memory[self.RM.get_index_cntr() + i] = item
            indices[i]=self.RM.get_index_cntr() + i
        self.RM.set(indices,new_prs)

    def sample_through(self):
        sampling_values = [0] * self.tbs
        for i in range(self.tbs):
            sampling_values[i]=(random.random() * 65)  # Generate a random float from 0 to sum_root (e.g., 65)
            batch = self.RM.get_prefix_sum_idx_sycl(sampling_values)
        states, actions, next_states, rewards, dones = \
        map(lambda x: torch.tensor(x).float(), zip(*batch))
        return states, actions, next_states, rewards, dones

    # new_prs have size train_bs
    def update_through(self, new_tds):
        self.RM.set(self.memoize_indices, new_tds)