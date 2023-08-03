import random
import torch
from replay_cgpu import SumTree,PrioritizedReplayMemory
from replay_cgpu import SumTreenary,PrioritizedReplayMemory_n
class TestData:
    def __init__(self, value):
        self.value = value

# Test code for ReplayMemory with SumTree
if __name__ == '__main__':

    # capacity = 27*9
    capacity = 2**9
    # fanout=3
    memory = PrioritizedReplayMemory(capacity)
    # memory = PrioritizedReplayMemory_n(capacity,fanout)

    intree_bs=2**2
    # intree_bs=3
    
    # Test add operation
    for i in range(capacity//intree_bs):
        # priority = random.random()*capacity*100
        priority = torch.rand(intree_bs)
        # Generate random data for testing
        random_data = [TestData(random.random()) for _ in range(intree_bs)]
        memory.push(random_data, priority)

    #test circular buffer. Success.
    random_data = [TestData(random.random()) for _ in range(intree_bs)]
    priority = torch.rand(intree_bs)
    memory.push(random_data, priority) 

    for i in range(1):
        # Test sample operation
        batch_size = 64
        batch, indexes= memory.sample(batch_size)
        assert (len(batch) == batch_size)
        assert (indexes.size(dim=0) == batch_size)

        # Test update operation
        td_errors = torch.rand(batch_size)
        # torch.rand(intree_bs)
        memory.update_priorities(indexes, td_errors)

    print("All operations (add, sample, update) with SumTree-based PrioritizedReplayMemory passed!")