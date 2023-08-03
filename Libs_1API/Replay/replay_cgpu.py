import random
import math
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# device = "cpu"

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("USING GPU")

'''
Uniform distribution
'''
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
        self.tree = torch.zeros(2 * capacity - 1, dtype=torch.float32).to(device)
        self.data_pointer = 0 #pointing to non-leaf and left-most leaf, range[0,capacity), corresponds to data storage dict indices. +capacity-1 to move to corresponding leaf node index

    def add(self, priorities):
        assert(len(list(priorities.size()))==1) #assert input tensor is an 1-D tensor
        intree_batchsize = priorities.size(dim=0)
        tree_indices = torch.zeros(intree_batchsize, dtype=torch.long)

        for i in range(intree_batchsize):
            tree_indices[i] = self.data_pointer + self.capacity - 1 + i
            # print("Sum Tree add at data pointer",self.data_pointer,"tree leaf index",tree_indices[i] )

        self.update(tree_indices, priorities)
        # print ("Sum Tree add done update")
        self.data_pointer = (self.data_pointer + intree_batchsize) % self.capacity #+1 in leaf, %cap to move back to non-leaf

    def update(self, tree_indices, priorities): # tree_index is given as the index in the tree, not index in the data storage dict. range: [capacity - 1, 2 * capacity - 1)
        # change = priority - self.tree[tree_index]
        assert(len(list(tree_indices.size()))==1) #assert input tensor is an 1-D tensor
        assert(len(list(priorities.size()))==1) #assert input tensor is an 1-D tensor
        assert(priorities.size(dim=0) == tree_indices.size(dim=0)) #assert input tensors have the same size
        intree_batchsize = priorities.size(dim=0)
        
        priorities = priorities.to(device)
        tree_indices = tree_indices.to(device)
        changes = torch.zeros(intree_batchsize, dtype=torch.float32).to(device)
        for i in range(intree_batchsize):
            changes[i] = priorities[i] - self.tree[tree_indices[i]]
            self.tree[tree_indices[i]] = priorities[i]

        # while tree_index != 0:
        #     tree_index = (tree_index - 1) // 2
        #     self.tree[tree_index] += change
        while not torch.all(tree_indices == 0):
            tree_indices = (tree_indices - 1) // 2
            for i in range(intree_batchsize):
                self.tree[tree_indices[i]] += changes[i]

    # used for sampling, return node indices in the tree (not data storage indices, needs conversion)
    def get_leaf(self, v):
        assert(len(list(v.size()))==1) #assert input tensor is an 1-D tensor
        intree_batchsize = v.size(dim=0)
        # parent_index = 0
        parent_indices = torch.zeros_like(v, dtype=torch.long, device=device)
        leaf_indices = torch.zeros_like(v, dtype=torch.long, device=device)
        leaf_priorities = torch.zeros_like(v, dtype=torch.float32, device=device)
        
        for i in range(intree_batchsize):
            while True:
                left_child_index = 2 * parent_indices[i] + 1
                right_child_index = left_child_index + 1

                if left_child_index >= len(self.tree): #reached leaf, no more leaf nodes
                    leaf_indices[i] = parent_indices[i]
                    leaf_priorities[i]=self.tree[leaf_indices[i]]
                    break
                else:
                    if v[i] <= self.tree[left_child_index]:
                        parent_indices[i] = left_child_index
                    else:
                        v[i] -= self.tree[left_child_index] #update target value
                        parent_indices[i] = right_child_index

        return leaf_indices, leaf_priorities

    def total_priority(self):
        return self.tree[0]

# Prioritized Replay based on a binary sum-tree
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
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

    def get_current_size(self):
        return self.current_size

    # for inserting a new experience
    def _get_priority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def push(self, transitions, td_errors):
        assert(type(transitions)==list)
        assert(len(transitions)==td_errors.size(dim=0)) #td_error is a 1D tensor
        intree_batchsize = len(transitions)
        priorities = self._get_priority(td_errors)
        self.tree.add(priorities)
        # self.data.append(transition)
        for i in range(intree_batchsize):
            self.data[self.data_pointer+i] = transitions[i]
        self.data_pointer = (self.data_pointer + intree_batchsize) % self.capacity
        if (self.current_size<self.capacity):
            self.current_size+=intree_batchsize
        # print("pushing data into data storage index",self.data_pointer)

    def sample(self,batch_size):
        batch = []
        segment = self.tree.total_priority() / batch_size
        priorities = []
        indexes = []
        v = torch.zeros(batch_size, dtype=torch.float32).to(device)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size): 
            a, b = segment * i, segment * (i + 1)
            v[i] = random.uniform(a, b)

        # get indices from sum tree
        indices, priorities = self.tree.get_leaf(v) #index returned are node index in the tree, not index in the data storage dict.

        # convert leaf ndoe index to data storage index, read from data storage sequntially
        ds_indices = (indices - self.tree.capacity + 1).to('cpu')
        for i in range(batch_size): 
            # device = 'cpu'
            batch.append(self.data[ds_indices[i].item()]) 
        assert(torch.all(ds_indices >= 0))
        assert(torch.all(ds_indices < self.tree.capacity))
        # print("sampling data with index:",index - self.tree.capacity + 1)

        # sampling_probabilities = priorities / self.tree.total_priority()
        # is_weights = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        # is_weights /= is_weights.max()

        # return batch, indexes, is_weights
        return batch, indices

    # index returned by sampling are node index in the tree, not index in the data storage dict.
    # Therefore, the same tree_indices (as returned by sampling) can be used for update without modification
    def update_priorities(self, tree_indices, td_errors):
        assert(len(list(tree_indices.size()))==1) #assert input tensor is an 1-D tensor
        assert(len(list(td_errors.size()))==1) #assert input tensor is an 1-D tensor
        # for tree_index, td_error in zip(tree_indices, td_errors):
        assert(torch.all(tree_indices >= self.tree.capacity - 1))
        assert(torch.all(tree_indices < 2*self.tree.capacity - 1))
        priorities = self._get_priority(td_errors)
        self.tree.update(tree_indices, priorities)




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
        print("num leaf nodes (capacity):",fanout**(i-1))

        self.num_non_leaf=full_tree_size-capacity
        # self.tree = {key: 0 for key in range(full_tree_size)}
        self.tree = torch.zeros(full_tree_size, dtype=torch.float32).to(device)

        self.data_pointer = 0


    def add(self, priorities):
        assert(len(list(priorities.size()))==1) #assert input tensor is an 1-D tensor
        intree_batchsize = priorities.size(dim=0)
        tree_indices = torch.zeros(intree_batchsize, dtype=torch.long)
        for i in range(intree_batchsize):
            tree_indices[i] = self.data_pointer + self.num_non_leaf + i
            # print("Sum Tree add at data pointer",self.data_pointer,"tree leaf index",tree_indices[i] )
        self.update(tree_indices, priorities)
        # print ("Sum Tree add done update")
        self.data_pointer = (self.data_pointer + intree_batchsize) % self.capacity #+1 in leaf, %cap to move back to non-leaf

    def update(self, tree_indices, priorities): # tree_index is given as the index in the tree, not index in the data storage dict. range: [capacity - 1, 2 * capacity - 1)
        # change = priority - self.tree[tree_index]
        assert(len(list(tree_indices.size()))==1) #assert input tensor is an 1-D tensor
        assert(len(list(priorities.size()))==1) #assert input tensor is an 1-D tensor
        assert(priorities.size(dim=0) == tree_indices.size(dim=0)) #assert input tensors have the same size
        intree_batchsize = priorities.size(dim=0)
        
        priorities = priorities.to(device)
        tree_indices = tree_indices.to(device)
        changes = torch.zeros(intree_batchsize, dtype=torch.float32).to(device)
        for i in range(intree_batchsize):
            changes[i] = priorities[i] - self.tree[tree_indices[i]]
            self.tree[tree_indices[i]] = priorities[i]

        while not torch.all(tree_indices == 0):
            tree_indices = (tree_indices - 1) // self.fanout
            for i in range(intree_batchsize):
                self.tree[tree_indices[i]] += changes[i]

    # used for sampling
    def get_leaf(self, v):
        assert(len(list(v.size()))==1) #assert input tensor is an 1-D tensor
        intree_batchsize = v.size(dim=0)
        # parent_index = 0
        parent_indices = torch.zeros_like(v, dtype=torch.long, device=device)
        leaf_indices = torch.zeros_like(v, dtype=torch.long, device=device)
        leaf_priorities = torch.zeros_like(v, dtype=torch.float32, device=device)
        
        for i in range(intree_batchsize):
            while True:
                left_child_index = self.fanout * parent_indices[i] + 1
                right_child_index = left_child_index + self.fanout-1

                if left_child_index >= len(self.tree): #reached leaf, no more leaf nodes
                    leaf_indices[i] = parent_indices[i]
                    leaf_priorities[i]=self.tree[leaf_indices[i]]
                    break
                else: 
                    psum = self.tree[left_child_index]
                    child_pointer = left_child_index
                    while (psum <= v[i]):
                        psum += self.tree[child_pointer]
                        if (child_pointer == right_child_index):
                            break
                        child_pointer += 1
                    v[i] -= self.tree[child_pointer]
                    parent_indices[i] = child_pointer
                    assert(child_pointer <= right_child_index)
                    # if v[i] <= self.tree[left_child_index]:
                    #     parent_indices[i] = left_child_index
                    # else:
                    #     v[i] -= self.tree[left_child_index] #update target value
                    #     parent_indices[i] = right_child_index

        return leaf_indices, leaf_priorities


    def total_priority(self):
        return self.tree[0]


# Prioritized Replay based on a n-ary sum-tree
class PrioritizedReplayMemory_n:
    def __init__(self, capacity, fanout, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.tree = SumTreenary(capacity, fanout)
        # self.data = deque(maxlen=capacity)
        self.data = {key: 0 for key in range(capacity)}
        self.data_pointer = 0
        self.epsilon = 0.01
        self.current_size = 0

    def get_current_size(self):
        return self.current_size

    # for inserting a new experience
    def _get_priority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def push(self, transitions, td_errors):
        assert(type(transitions)==list)
        assert(len(transitions)==td_errors.size(dim=0)) #td_error is a 1D tensor
        intree_batchsize = len(transitions)
        priorities = self._get_priority(td_errors)
        self.tree.add(priorities)
        # self.data.append(transition)
        for i in range(intree_batchsize):
            self.data[self.data_pointer+i] = transitions[i]
        self.data_pointer = (self.data_pointer + intree_batchsize) % self.capacity
        if (self.current_size<self.capacity):
            self.current_size+=intree_batchsize
        # print("pushing data into data storage index",self.data_pointer)

    def sample(self,batch_size):
        batch = []
        segment = self.tree.total_priority() / batch_size
        priorities = []
        indexes = []
        v = torch.zeros(batch_size, dtype=torch.float32).to(device)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size): 
            a, b = segment * i, segment * (i + 1)
            v[i] = random.uniform(a, b)

        # get indices from sum tree
        indices, priorities = self.tree.get_leaf(v) #index returned are node index in the tree, not index in the data storage dict.

        # convert leaf ndoe index to data storage index, read from data storage sequntially
        ds_indices = (indices - self.tree.num_non_leaf).to('cpu')
        for i in range(batch_size): 
            # device = 'cpu'
            batch.append(self.data[ds_indices[i].item()]) 
        assert(torch.all(ds_indices >= 0))
        assert(torch.all(ds_indices < self.tree.capacity))
        # print("sampling data with index:",index - self.tree.capacity + 1)

        # sampling_probabilities = priorities / self.tree.total_priority()
        # is_weights = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        # is_weights /= is_weights.max()

        # return batch, indexes, is_weights
        return batch, indices

    # index returned by sampling are node index in the tree, not index in the data storage dict.
    # Therefore, the same tree_indices (as returned by sampling) can be used for update without modification
    def update_priorities(self, tree_indices, td_errors):
        assert(len(list(tree_indices.size()))==1) #assert input tensor is an 1-D tensor
        assert(len(list(td_errors.size()))==1) #assert input tensor is an 1-D tensor
        # for tree_index, td_error in zip(tree_indices, td_errors):
        assert(torch.all(tree_indices >= self.tree.num_non_leaf))
        assert(torch.all(tree_indices < 2*self.tree.full_tree_size))
        priorities = self._get_priority(td_errors)
        self.tree.update(tree_indices, priorities)

