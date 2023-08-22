
from sycl_rm_module import SumTreeNary
import random

PTree = SumTreeNary(1024, 16)

for i in range(128):
    idx_vect = [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3]
    val_vect = [0.1 * i * 4, 0.1 * (i * 4 + 1), 0.1 * (i * 4 + 2), 0.1 * (i * 4 + 3)]
    PTree.set(idx_vect, val_vect); 


sampled_ind = [0] * 8  # Initializing a list of size 8 for sampled indices
sampling_values = [0] * 8

for i in range(8):
    sampling_values[i]=(random.random() * 65)  # Generate a random float from 0 to 65

print("Created emulated sampling values:", sampling_values[0], "to", sampling_values[7])

sampled_ind = PTree.get_prefix_sum_idx_sycl(sampling_values)

print("sampled indices:", sampled_ind[0], sampled_ind[1], sampled_ind[2],
      sampled_ind[3], sampled_ind[4], sampled_ind[5], sampled_ind[6], sampled_ind[7])
