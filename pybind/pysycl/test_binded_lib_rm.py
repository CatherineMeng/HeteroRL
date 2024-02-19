
from sycl_rm_module import SumTreeNary
import random
import time 


upd_bs=256
samp_bs=512

PTree = SumTreeNary(1024, 16)

start = time.perf_counter()
idx_vect=[0]*upd_bs
val_vect=[0]*upd_bs
for i in range(upd_bs):
    idx_vect[i] = i
    val_vect[i] = 0.1*(i*4+i%4)
PTree.set(idx_vect, val_vect); 
print("Update of batch size",upd_bs,"took",(time.perf_counter()-start)* 2000,"ms")


sampled_ind = [0] * samp_bs  # Initializing a list of size 8 for sampled indices
sampling_values = [0] * samp_bs

for i in range(samp_bs):
    sampling_values[i]=(random.random() * 65)  # Generate a random float from 0 to 65

# print("Created emulated sampling values:", sampling_values[0], "to", sampling_values[7])
start = time.perf_counter()
sampled_ind = PTree.get_prefix_sum_idx_sycl(sampling_values)
# print("Sampling of batch size",samp_bs,"took",(time.perf_counter()-start)* 1000,"ms")
# print("sampled indices:", sampled_ind[0], sampled_ind[1], sampled_ind[2],
#       sampled_ind[3], sampled_ind[4], sampled_ind[5], sampled_ind[6], sampled_ind[7])
