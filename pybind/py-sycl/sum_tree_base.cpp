#include "sum_tree_base.h"


// auto SumTree::sample_idx(int64_t batch_size) const -> torch::Tensor {
//     auto scalar = torch::rand({batch_size}) * reduce();
//     auto idx = this->get_prefix_sum_idx(scalar);
//     return idx;
// }