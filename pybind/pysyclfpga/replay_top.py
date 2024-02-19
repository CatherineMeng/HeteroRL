from replay_module import PER
from replay_module import sibit_io
import random
import torch

# n-ary sum tree
# check consistency for K, D in replay_cpplib.cpps
class SumTreenary_FPGA:
    def __init__(self, depth, fanout, train_bs, insert_bs):
        self.D = depth
        self.K = fanout
        self.replay_manager = PER()

        self.root_pr=0
        # replay_manager.Test(root_pr)
        self.replay_manager.Init(root_pr)
        self.root_sum_val = 0

        self.tbs=train_bss
        self.ibs=insert_bs

        self.in_sample_list = [sibit_io()]*train_bs
        self.out_sampled_idx = [0]*train_bs
        self.out_sampled_value = [0]*train_bs
        self.in_update_list = [sibit_io()]*train_bs
        #runtime re-allocation of actor threads change the rate of buffer population, but not insert_bs
        self.in_insertion_list = [sibit_io()]*insert_bs 
        self.out_insertion_getPr_value = [0]*insert_bs

        self.insert_indcntr = 0 #keep track of the current insertion position, fifo eviction

    def get_index_cntr(self):
        return self.insert_indcntr

    # Sampling - based on train_bs.
    # depending on learner_device, output format can be tensor (for cpu/gpu parsing) or list (for fpga parsing)
    def sample(self,l_device):
        for i in range(self.tbs):
            self.in_sample_list[i].sampling_flag=1
            self.in_sample_list[i].update_flag=0
            self.in_sample_list[i].get_priority_flag=0
            self.in_sample_list[i].init_flag=0
            self.in_sample_list[i].start=0
            self.in_sample_list[i].newx=random.randint(0, self.root_sum_val-1)
        MultiKernel_out = replay_manager.DoWorkMultiKernel(self.in_sample_list, self.out_sampled_idx, self.out_sampled_value, self.out_insertion_getPr_value,\
        self.root_pr, self.tbs, 1)
        leaf_indices = [None]*self.tbs
        for i in range(self.tbs):
            leaf_indices[i]=MultiKernel_out.out_pr_sampled[i]
        self.out_sampled_idx = leaf_indices
        return leaf_indices
        
    # priorities is a 1D list/tensor of size insert_batch_size
    # do not need an extra list of indices - memoized during sampling. 
    def update(self, priorities_tb):
        for i in range(self.ibs):
            self.in_update_list[i].sampling_flag=0
            self.in_update_list[i].update_flag=1
            self.in_update_list[i].get_priority_flag=0
            self.in_update_list[i].init_flag=0
            self.in_update_list[i].update_index_array[0]=0
            self.in_update_list[i].update_index_array[self.D-1]=(self.out_sampled_idx[i])
            ii=self.D-2
            while (ii<=self.D):
                self.in_update_list[i].update_index_array[ii]=(self.in_update_list[i].update_index_array[ii+1])/K
            # in_list[i].update_index_array[1]=((self.insert_indcntr+i)/K)/K
            # in_list[i].update_index_array[2]=(self.insert_indcntr+i)/K
            # in_list[i].update_index_array[3]=(self.insert_indcntr+i)

            for ii in range(D):
                self.in_update_list[i].update_offset_array[ii] = priorities_tb[i]

        MultiKernel_out = replay_manager.DoWorkMultiKernel(self.in_update_list, self.out_sampled_idx, self.out_sampled_value, self.out_insertion_getPr_value,\
        self.root_pr, self.tbs, 1)


    # priorities is a 1D list/tensor of size insert_batch_size
    def insert(self, priorities):
        if (self.insert_indcntr>memory_size-self.ibs): #avoid overflow
            self.insert_indcntr=0
        
        #get priority
        for i in range(self.ibs):
            self.in_insertion_list[i].sampling_flag=0
            self.in_insertion_list[i].update_flag=0
            self.in_insertion_list[i].get_priority_flag=1
            self.in_insertion_list[i].init_flag=0
            self.in_insertion_list[i].pr_idx=self.insert_indcntr+i

        MultiKernel_out = replay_manager.DoWorkMultiKernel(self.in_insertion_list, self.out_sampled_idx, self.out_sampled_value, self.out_insertion_getPr_value,\
        self.root_pr, self.ibs, 1)
        # results in self.out_insertion_getPr_value
        
        # Insertion -> Update
        for i in range(self.ibs):
            self.in_insertion_list[i].sampling_flag=0
            self.in_insertion_list[i].update_flag=1
            self.in_insertion_list[i].get_priority_flag=0
            self.in_insertion_list[i].init_flag=0
            self.in_insertion_list[i].update_index_array[0]=0
            self.in_insertion_list[i].update_index_array[self.D-1]=(self.insert_indcntr+i)
            ii=self.D-2
            while (ii<=self.D):
                self.in_insertion_list[i].update_index_array[ii]=(self.in_insertion_list[i].update_index_array[ii+1])/K
            # in_list[i].update_index_array[1]=((self.insert_indcntr+i)/K)/K
            # in_list[i].update_index_array[2]=(self.insert_indcntr+i)/K
            # in_list[i].update_index_array[3]=(self.insert_indcntr+i)

            for ii in range(D):
                self.in_insertion_list[i].update_offset_array[ii] = self.out_insertion_getPr_value[i]-priorities[i]

        print("=== Running the update kernel ===")
        MultiKernel_out = replay_manager.DoWorkMultiKernel(self.in_insertion_list,  self.out_sampled_idx,  self.out_sampled_value,  self.out_insertion_getPr_value,\
        self.root_pr, self.ibs, 1)

        # increment self.insert_indcntr by ins bacth size for next-round insertion
        self.insert_indcntr += self.ibs
        
    def total_priority(self):
        return self.tree[0]

class replay_top():
    def __init__(self, depth, fanout, train_bs, insert_bs, memory_size=10000):
        self.RM = SumTreenary_FPGA(depth, fanout, train_bs, insert_bs)
        self.memory = {key: None for key in range(memory_size)} #data storage
        self.memory_size = memory_size
        

    def __len__(self):
        return len(self.memory)

    # in both sum tree and data storage
    # indices, items and new_prs have size insert_bs
    # each item will follow the format [state, action, next_state, reward, done]
    def insert_through(self, items, new_prs):
        for i,item in enumerate(items):
            self.memory[self.RM.get_index_cntr() + i] = item
        self.RM.insert(new_prs)

    def sample_through(self):
        states, actions, next_states, rewards, dones = \
        map(lambda x: torch.tensor(x).float(), zip(*self.RM.sample(" ")))
        return states, actions, next_states, rewards, dones

    # new_prs have size train_bs
    def update_through(self, new_tds):
        self.RM.insert(new_tds)