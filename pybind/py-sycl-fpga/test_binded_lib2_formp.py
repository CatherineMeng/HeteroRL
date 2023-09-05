# This is used as a priliminary test for use with mp runtime
# Once succeed, integrate into mp_train_AL_pybRfpga.py
from replay_module import PER
from replay_module import sibit_io

import gym
import torch
import random
import numpy as np

def cliprg(x,lo,hi):
    if (x<lo):
        return lo
    elif (x>=hi):
        return hi-1
    else:
        return x
# n-ary sum tree - sycl FPGA pybinded module
class SumTreenary_fpga:
    def __init__(self, capacity, fanout):
        self.capacity = capacity
        self.fanout = fanout
        # assume capacity is a power of fanout!
        i=0
        full_tree_size=0
        while (fanout**i<=capacity):
            full_tree_size+=fanout**i
            i+=1
        self.full_tree_size=full_tree_size
        # self.full_tree_size - self.capacity = capacity is the total number of non-leaf nodes
        assert(self.capacity==fanout**(i-1))
        self.num_non_leaf=full_tree_size-capacity
        self.D = 4 #depth
        self.tree = PER()
        self.root_pr=0
        # replay_manager.Test(root_pr)
        self.tree.Init(self.root_pr)
        self.data_pointer = 0

    def add(self, priorities, BS):
        assert(len(priorities)==BS)
        inds = [self.data_pointer+i for i in range(BS)]
        self.update(inds, priorities, BS)
        self.data_pointer = (self.data_pointer + BS) % self.capacity

    def update(self, inds, priorities, BS):
        assert(len(priorities)==BS)
        in_list = [sibit_io()]*BS
        for ii in range(BS):
            in_list[ii].sampling_flag=0
            in_list[ii].update_flag=0
            in_list[ii].get_priority_flag=1
            in_list[ii].init_flag=0
            in_list[ii].pr_idx=inds[ii]
            self.root_pr += priorities[ii]
        # print("=== Running the get-priority kernel ===")
        MultiKernel_out = self.tree.DoWorkMultiKernel(in_list, [0]*BS, [0.0]*BS, [0.0]*BS, \
        self.root_pr, BS, 1)
        delts = [priorities[i] - MultiKernel_out.out_pr_insertion[i] for i in range(BS)]

        for ii in range(BS):
            in_list[ii].sampling_flag=0
            in_list[ii].update_flag=1
            in_list[ii].get_priority_flag=0
            in_list[ii].init_flag=0
            in_list[ii].update_index_array[0]=0
            in_list[ii].update_index_array[1]=(ii/self.fanout)/self.fanout
            in_list[ii].update_index_array[2]=ii/self.fanout
            in_list[ii].update_index_array[3]=ii
            for iii in range(self.D):
                in_list[ii].update_offset_array[iii]=delts[ii]
                # in_list[ii].set_upd_offset_index(iii,0.1)
        # print("=== Running the update kernel ===")
        MultiKernel_out = self.tree.DoWorkMultiKernel(in_list, [0]*BS, [0.0]*BS, [0.0]*BS,\
        self.root_pr, BS, 1)
   
    # used for sampling
    def get_leaf(self, vs, BS):
        assert(len(vs)==BS)
        in_list = [sibit_io()]*BS
        for ii in range(BS):
            in_list[ii].sampling_flag=1
            in_list[ii].update_flag=0
            in_list[ii].get_priority_flag=0
            in_list[ii].init_flag=0
            in_list[ii].start=0
            in_list[ii].newx=vs[ii]
        # print("=== Running the Sampling kernel ===")
        MultiKernel_out = self.tree.DoWorkMultiKernel(in_list, [0]*BS, [0.0]*BS, [0.0]*BS, \
        self.root_pr, BS, 1)
        return MultiKernel_out.sampled_idx, MultiKernel_out.out_pr_sampled
        
    def total_priority(self):
        return self.root_pr


# Prioritized Replay based on an FPGA-implemented n-ary sum-tree 
class PrioritizedReplayMemory_n_fpga:
    def __init__(self, capacity,fanout, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.tree = SumTreenary_fpga(capacity,fanout)
        # self.data = deque(maxlen=capacity)
        self.data = {key: 0 for key in range(capacity)}
        self.data_pointer = 0
        self.epsilon = 0.01

    # for inserting a new experience
    def _get_priority(self, td_error):
        return [(tde + self.epsilon) ** self.alpha for tde in td_error]

    # changed to batched mode
    def push(self, transitions, td_errors, insert_batchsize):
        assert(len(transitions)==insert_batchsize)
        assert(len(td_errors)==insert_batchsize)
        prs = self._get_priority(td_errors)
        self.tree.add(prs,insert_batchsize)
        for i in range(self.data_pointer, self.data_pointer+insert_batchsize):
            self.data[i] = transitions[i-self.data_pointer]
            # print("pushing data into data storage index",i)
        self.data_pointer = (self.data_pointer + insert_batchsize) % self.capacity
        

    def sample(self, batch_size):
        segment = self.tree.total_priority() / batch_size
        vs = [random.uniform(segment * i, segment * (i+1))*self.tree.total_priority() for i in range(batch_size)]
        indices, priorities = self.tree.get_leaf(vs,batch_size) #indices returned are indices in the data storage dict.
        indices_c = [cliprg(x,0,self.capacity) for x in indices]
        batch = [self.data[indices_c[i]] for i in range(batch_size)]
        return batch, indices

    def update_priorities(self, tree_indices, td_errors):
        assert(len(tree_indices)==len(td_errors))
        self.tree.update(tree_indices,td_errors,len(tree_indices))


if __name__ == '__main__':
    capacity = 4**3
    insert_BS=16
    train_BS=8
    fanout=4
    replay_memory = PrioritizedReplayMemory_n_fpga(capacity,fanout)

    env = gym.make('CartPole-v1')
    exp_buffer=[None]*insert_BS
    td_errors=[None]*insert_BS
    exp_buffer_cnter=0
    while (exp_buffer_cnter<insert_BS):
        state = env.reset()[0]
        done = False
        while not done:
            action = np.random.randint(0, 2)
            next_state, reward, done, _, _ = env.step(action)
            exp_buffer[exp_buffer_cnter]=(state, action, reward, next_state, done)
            td_errors[exp_buffer_cnter]= np.random.uniform(0.0, 1.0)
            state = next_state
            exp_buffer_cnter+=1
            if (exp_buffer_cnter==insert_BS):
                break

    print("Testing buffered push")
    replay_memory.push(exp_buffer,td_errors,insert_BS)

    print("Testing bacthed sampling")
    transitions, sampled_indices = replay_memory.sample(train_BS)

    print("Testing bacthed update")
    td_errors_train = [random.random() for _ in range(train_BS)]
    replay_memory.update_priorities(sampled_indices, td_errors_train)

    print("done.")