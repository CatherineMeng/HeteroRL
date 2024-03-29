import random
import math
from collections import namedtuple, deque
import torch
'''
Uniform distribution
'''

class Memory(object):
  def __init__(self, memory_size, train_bs, insert_bs):
    self.memory = deque(maxlen=memory_size)
    self.memory_size = memory_size
    self.train_bs = train_bs
    self.insert_bs = insert_bs 

  def __len__(self):
    return len(self.memory)

  def append(self, item):
    self.memory.append(item)

  def sample_batch(self, batch_size):
    idx = np.random.permutation(len(self.memory))[:batch_size]
    return [self.memory[i] for i in idx]

class ReplayMemory(object):

    def __init__(self, capacity, train_bs, insert_bs):
        self.memory = deque([], maxlen=capacity)
        self.train_bs = train_bs
        self.insert_bs = insert_bs 

    def get_current_size(self):
        return len(self.memory)

    def push(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def insert_through(self, transitions, td_error):
        for tn in transitions:
            self.memory.append(tn)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_through(self):
        return random.sample(self.memory, self.train_bs)

    def update_through(self, prs):
        return 0

    def __len__(self):
        return len(self.memory)

import random
from collections import deque

def Log2(x):
    return (math.log10(x) / math.log10(2));
 
def n_isPowerOf_f(n,f):
    if (n == 0):
        return False
    while (n != 1):
        if (n % f != 0):
            return False
        n = n // f
    return True

# binary sum-tree
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # assume capacity is a power of 2!
        assert(math.ceil(Log2(capacity)) == math.floor(Log2(capacity)))
        # self.tree = [0] * (2 * capacity - 1)
        self.tree = {key: 0 for key in range(2 * capacity - 1)}
        self.data_pointer = 0 #pointing to non-leaf and left-most leaf, range[0,capacity), corresponds to data storage dict indices. +capacity-1 to move to corresponding leaf node index

    def add(self, priority):
        tree_index = self.data_pointer + self.capacity - 1
        # print("Sum Tree add at data pointer",self.data_pointer,"tree leaf index",self.data_pointer + self.capacity - 1)
        self.update(tree_index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity #+1 in leaf, %cap to move back to non-leaf

    def update(self, tree_index, priority): # tree_index is given as the index in the tree, not index in the data storage dict. range: [capacity - 1, 2 * capacity - 1)
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    # used for sampling, return node indices in the tree (not data storage indices, needs conversion)
    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree): #reached leaf, no more leaf nodes
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index] #update target value
                    parent_index = right_child_index

        return leaf_index, self.tree[leaf_index]

    def total_priority(self):
        return self.tree[0]

# Prioritized Replay based on a binary sum-tree
class PrioritizedReplayMemory:
    def __init__(self, capacity, train_bs, insert_bs, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.tree = SumTree(capacity)
        # self.data = deque(maxlen=capacity)
        self.data = {key: 0 for key in range(capacity)}
        self.data_pointer = 0
        self.epsilon = 0.01
        self.current_size = 0
        self.tbs = train_bs
        self.ibs = insert_bs
        self.memoize_ind = []

    def get_current_size(self):
        return self.current_size

    # for inserting a new experience
    def _get_priority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def insert_through(self, transition, td_error):
        for tn,te in zip(transition, td_error):
            # priority = self._get_priority(td_error)
            priority = te
            self.tree.add(priority)
            # self.data.append(tn)
            self.data[self.data_pointer] = tn
            self.data_pointer = (self.data_pointer + 1) % self.capacity
            if (self.current_size<self.capacity):
                self.current_size+=1
            # print("pushing data into data storage index",self.data_pointer)

    def sample_through(self):
        batch = []
        segment = self.tree.total_priority() / self.tbs
        priorities = []

        # indexes = []

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(self.tbs):
            a, b = segment * i, segment * (i + 1)
            v = random.uniform(a, b)
            # get indices from sum tree
            index, priority = self.tree.get_leaf(v) #index returned are node index in the tree, not index in the data storage dict.
            self.memoize_ind.append(index)
            priorities.append(priority)
            # convert leaf ndoe index to data storage index, read from data storage
            batch.append(self.data[index - self.tree.capacity + 1]) 
            assert(index - self.tree.capacity + 1 >= 0)
            assert(index - self.tree.capacity + 1 < self.tree.capacity)
            # print("sampling data with index:",index - self.tree.capacity + 1)

        # sampling_probabilities = priorities / self.tree.total_priority()
        # is_weights = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        # is_weights /= is_weights.max()
        # states, actions, next_states, rewards, dones = \
        #     map(lambda x: torch.tensor(x).float(), zip(*batch))
        return batch
        # return states, actions, next_states, rewards, dones

    # index returned by sampling are node index in the tree, not index in the data storage dict.
    # Therefore, the same tree_indices (as returned by sampling) can be used for update without modification
    def update_through(self, td_errors):
        for tree_index, td_error in zip(self.memoize_ind, td_errors):
            assert(tree_index >= self.tree.capacity - 1)
            assert(tree_index < 2*self.tree.capacity - 1)
            priority = self._get_priority(td_error)
            self.tree.update(tree_index, priority)




# n-ary sum tree
class SumTreenary:
    def __init__(self, capacity, fanout):
        self.capacity = capacity
        self.fanout = fanout
        # self.tree = [0] * (2 * capacity - 1)
        # assume capacity is a power of fanout!
        assert(n_isPowerOf_f(capacity,fanout))
        i=0
        full_tree_size=0
        while (fanout**i<=capacity):
            full_tree_size+=fanout**i
            i+=1
        print("full tree size:",full_tree_size)
        self.full_tree_size=full_tree_size
        # self.full_tree_size - self.capacity = capacity is the total number of non-leaf nodes
        assert(self.capacity==fanout**(i-1))
        print("num leaf nodes:",fanout**(i-1))
        self.num_non_leaf=full_tree_size-capacity
    
        self.tree = {key: 0 for key in range(full_tree_size)}

        self.data_pointer = 0

    def add(self, priority):
        tree_index = self.data_pointer + self.num_non_leaf
        self.update(tree_index, priority)
        print("insertion updating tree index:",tree_index)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
    def update(self, tree_index, priority):
        
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // self.fanout
            self.tree[tree_index] += change

    # used for sampling
    def get_leaf(self, v):
        parent_index = 0

        # level=0 #for debugging
        while True:
            
            left_child_index = self.fanout * parent_index + 1
            right_child_index = left_child_index + self.fanout-1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: 
                psum = self.tree[left_child_index]
                child_pointer = left_child_index
                # print("level",level,"starting at index",child_pointer)#for debugging
                while (psum <= v):
                    psum += self.tree[child_pointer]
                    if (child_pointer == right_child_index):
                        break
                    child_pointer += 1
                    # print("level",level,"moving to index",child_pointer) #for debugging
                v -= self.tree[child_pointer]
                parent_index = child_pointer
                assert(child_pointer <= right_child_index)
            # level+=1 #for debugging
        # returned leaf_index is tree node index, not data storage index, so caan be directly used for update
        return leaf_index, self.tree[leaf_index]
        

    def total_priority(self):
        return self.tree[0]


# Prioritized Replay based on a n-ary sum-tree
class PrioritizedReplayMemory_n:
    def __init__(self, capacity,fanout, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.tree = SumTreenary(capacity,fanout)
        # self.data = deque(maxlen=capacity)
        self.data = {key: 0 for key in range(capacity)}
        self.data_pointer = 0
        self.epsilon = 0.01

    # for inserting a new experience
    def _get_priority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def push(self, transition, td_error):
        priority = self._get_priority(td_error)
        self.tree.add(priority)
        # self.data.append(transition)
        self.data[self.data_pointer] = transition
        # print("pushing data into data storage index",self.data_pointer)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        

    def sample(self, batch_size):
        batch = []
        segment = self.tree.total_priority() / batch_size
        print("self.tree.total_priority():",self.tree.total_priority(),"; segment:",segment)
        priorities = []
        indexes = []

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            v = random.uniform(a, b)*self.tree.total_priority()
            # get indices from sum tree
            index, priority = self.tree.get_leaf(v) #index returned are node index in the tree, not index in the data storage dict.
            indexes.append(index)
            priorities.append(priority)
            # convert leaf ndoe index to data storage index, for reading from data storage
            batch.append(self.data[index - (self.tree.full_tree_size - self.tree.capacity)]) 
            # assertions: sampled index is a tree leaf index
            assert(index >= self.tree.full_tree_size - self.tree.capacity)
            assert(index < self.tree.full_tree_size)
            # print("sampling data with index:",index - (self.tree.full_tree_size - self.tree.capacity))

        return batch, indexes

    # index returned by sampling are node index in the tree, not index in the data storage dict.
    # Therefore, the same tree_indices (as returned by sampling) can be used for update without modification
    def update_priorities(self, tree_indices, td_errors):
        for tree_index, td_error in zip(tree_indices, td_errors):
            # assertions: updated index is a tree leaf index
            assert(tree_index >= self.tree.full_tree_size - self.tree.capacity)
            assert(tree_index < self.tree.full_tree_size)
            priority = self._get_priority(td_error)
            self.tree.update(tree_index, priority)


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


