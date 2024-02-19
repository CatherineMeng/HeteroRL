
from sycl_learner_module import DQNTrainer
from sycl_learner_module import params_out

import time

BS = 512
L1 = 8
L2 = 128
L3 = 128
L4 = 4



hw1 = [[(i + 1.0) / 2 for j in range(L2)] for i in range(L1)]
hb1 = [0.1 for _ in range(L2)]
hw2 = [[(i + 1.0) / 2 for j in range(L3)] for i in range(L2)]
hb2 = [0.1 for _ in range(L3)]
ow = [[(i + 1.0) for j in range(L4)] for i in range(L3)]
ob = [0.2 for _ in range(L4)]

trainer = DQNTrainer(hw1, hb1, hw2, hb2, ow, ob)

inputs = [[0.1 for j in range(L1)] for i in range(BS)]
snt_inputs = [[0.5 for j in range(L1)] for i in range(BS)]

rs = [1.5 for _ in range(BS)]
ds = [0 for _ in range(BS)]
as_ = [0 for _ in range(BS)]
for i in range(BS):
    if (i==0):
        as_[i]=0
    else:
        if (as_[i-1] == 0):
            as_[i]=1
        else:
            as_[i]=0

start=time.perf_counter()
for i in range(100):
    new_prs = trainer.train_itr(inputs, as_, snt_inputs, rs, ds, False)
    # print("new_prs:", new_prs)
    trainer.train_itr(inputs, as_, snt_inputs, rs, ds, True)
end=time.perf_counter()
print("One training itr batch of", BS, ", exe time:",(end-start)/200*1000,"ms")

# dpack = trainer.updated_params()
# dpack.print_params()