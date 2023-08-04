import random
import time
import torch
from replay_cgpu import SumTree,PrioritizedReplayMemory
from replay_cgpu import SumTreenary,PrioritizedReplayMemory_n
class TestData:
    def __init__(self, value):
        self.value = value

# Test code for ReplayMemory with SumTree
if __name__ == '__main__':

    # capacity = 27*9
    capacity = 16**3
    fanout=16
    memory = PrioritizedReplayMemory(capacity)
    # memory = PrioritizedReplayMemory_n(capacity,fanout)

    intree_bs=32
    # intree_bs=3
    
    t_start=time.perf_counter()
    # Test add operation
    for i in range(capacity//intree_bs):
        # priority = random.random()*capacity*100
        priority = torch.rand(intree_bs)
        # Generate random data for testing
        random_data = [TestData(random.random()) for _ in range(intree_bs)]
        memory.push(random_data, priority)
    t_end=time.perf_counter()
    print("per-batch insertion time:",(t_end - t_start)/(capacity//intree_bs))

    #test circular buffer. Success.
    random_data = [TestData(random.random()) for _ in range(intree_bs)]
    priority = torch.rand(intree_bs)
    memory.push(random_data, priority) 

    
    t_sample_total=0
    t_update_total=0
    for i in range(100):
        # Test sample operation
        batch_size = intree_bs
        t_start=time.perf_counter()
        batch, indexes= memory.sample(batch_size)
        assert (len(batch) == batch_size)
        assert (indexes.size(dim=0) == batch_size)
        t_end=time.perf_counter()
        t_sample_total += t_end-t_start

        t_start=time.perf_counter()
        # Test update operation
        td_errors = torch.rand(batch_size)
        # torch.rand(intree_bs)
        memory.update_priorities(indexes, td_errors)
        t_end=time.perf_counter()
        t_update_total += t_end-t_start
    
    print("per-batch sample time:",t_sample_total/100)
    print("per-batch update time:",t_update_total/100)

    print("All operations (add, sample, update) with SumTree-based PrioritizedReplayMemory passed!")