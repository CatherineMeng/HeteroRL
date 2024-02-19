
from sycl_learner_module import DQNTrainer
from sycl_learner_module import params_out

L1 = 4
L2 = 8
L3 = 2
BS = 8


hw = [[(i + 1.0) / 2 for j in range(L2)] for i in range(L1)]
hb = [0.1 for _ in range(L2)]
ow = [[(i + 1.0) for j in range(L3)] for i in range(L2)]
ob = [0.2 for _ in range(L3)]

trainer = DQNTrainer(hw, hb, ow, ob)

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

new_prs = trainer.train_itr(inputs, as_, snt_inputs, rs, ds, False)
print("new_prs:", new_prs)

trainer.train_itr(inputs, as_, snt_inputs, rs, ds, True)
dpack = trainer.updated_params()
dpack.print_params()