
from sycl_rm_module import SumTreeNary
import time 
import random
import torch


class replay_top():
    def __init__(self, fanout, train_bs, insert_bs, memory_size=8000):
        self.RM = SumTreeNary(memory_size, fanout) 
        self.memory = {key: None for key in range(memory_size)} #data storage
        self.memory_size = memory_size
        self.ibs = insert_bs
        self.tbs = train_bs
        self.memoize_indices = [None]*train_bs
        self.curr_ind = 0
        

    def __len__(self):
        return len(self.memory)

    def get_current_size(self):
        return self.curr_ind

    # in both sum tree and data storage
    # indices, items and new_prs have size insert_bs
    # each item will follow the format [state, action, next_state, reward, done]
    def insert_through(self, items, new_prs):
        indices = [None] * self.ibs

        for i,item in enumerate(items):
            self.memory[self.curr_ind + i] = item
            indices[i]=self.curr_ind + i
        self.RM.set(indices,new_prs)
        if (self.curr_ind + self.ibs < self.memory_size):
            self.curr_ind += self.ibs
        # else:
        #     self.curr_ind =0

    def sample_through(self):
        batch = []
        sampling_values = [0] * self.tbs
        for i in range(self.tbs):
            sampling_values[i]=(random.random() * 65)  # Generate a random float from 0 to sum_root (e.g., 65)
        batch_ind = self.RM.get_prefix_sum_idx_sycl(sampling_values)
        for i in range(self.tbs):
            self.memoize_indices[i] = batch_ind[i]
            batch.append(self.memory[batch_ind[i]])
        # states, actions, next_states, rewards, dones = map(lambda x: torch.tensor(x).float(), zip(*batch))
        return batch

    # new_prs have size train_bs
    def update_through(self, new_tds):
        self.RM.set(self.memoize_indices, new_tds)


# ====================================================
# === Synthetic Test for Runtime Program Integration
# ====================================================
insert_bs = 128
train_bs = 128

in_dim = 4
out_dim = 2


PER_sycl = replay_top(16,train_bs,insert_bs)

val_vect=[0]*insert_bs
items=[None]*insert_bs
start = time.perf_counter()
for j in range(int(10000/insert_bs)):
    for i in range(insert_bs):
        val_vect[i] = 0.1*(i*4+i%4)
        state = torch.randn(in_dim)
        next_state = torch.randn(in_dim)
        action = torch.randn(out_dim)
        reward = random.random()
        done = random.choice([True, False])
        items[i] = [state, action, next_state, reward, done]
    PER_sycl.insert_through(items, val_vect); 
print("Inserting",10000,"samples took",(time.perf_counter()-start)* 1000,"ms (synthetic functional behavior of Actor processes)")


sampled_ind = [0] * train_bs  # Initializing a list of size 8 for sampled indices
sampling_values = [0] * train_bs
for i in range(train_bs):
    sampling_values[i]=(random.random() * 65)  # Generate a random float from 0 to 65
# print("Created emulated sampling values:", sampling_values[0], "to", sampling_values[7])
start = time.perf_counter()
sampled_data = PER_sycl.sample_through()
# print(sampled_data)
# print("Sampling of batch size",insert_bs,"took",(time.perf_counter()-start)* 1000,"ms")

val_vect=[0]*insert_bs
for i in range(insert_bs):
    val_vect[i] = 0.3*(i*4+i%4)
start = time.perf_counter()
PER_sycl.update_through(val_vect); 
print("Update of batch size",insert_bs,"took",(time.perf_counter()-start)* 1000,"ms")

