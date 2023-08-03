import random
from replay import SumTree,PrioritizedReplayMemory
from replay import SumTreenary,PrioritizedReplayMemory_n
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

    # Generate random data for testing
    random_data = [TestData(random.random()) for _ in range(capacity)]

    # Test add operation
    for i in range(capacity):
        priority = random.random()*capacity*100
        memory.push(random_data[i], priority)
    memory.push(random_data[i], priority) #test circular buffer. yes

    # Test sample operation
    # batch_size = 64
    batch_size = 16
    batch, indexes= memory.sample(batch_size)
    assert len(batch) == batch_size
    assert len(indexes) == batch_size

    # Test update operation
    td_errors = [random.random() for _ in range(batch_size)]
    memory.update_priorities(indexes, td_errors)

    print("All operations (add, sample, update) with SumTree-based PrioritizedReplayMemory passed!")