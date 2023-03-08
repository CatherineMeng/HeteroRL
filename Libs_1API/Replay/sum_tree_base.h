#ifndef SUM_TREE_BASE_H
#define SUM_TREE_BASE_H

#include <torch/torch.h>
#include <vector>


class SumTree {
public:
    [[nodiscard]] virtual auto size() const -> int64_t = 0;

    virtual auto operator[](const torch::Tensor &idx) const -> torch::Tensor = 0;

    virtual auto set(const torch::Tensor &idx, const torch::Tensor &value) -> void = 0;

    [[nodiscard]] virtual auto reduce(int64_t start, int64_t end) const -> float = 0;

    [[nodiscard]] virtual auto reduce() const -> float = 0;

    [[nodiscard]] virtual auto get_prefix_sum_idx(torch::Tensor value) const -> torch::Tensor = 0;

    // for compatible with the abstraction of the FPGA segment tree
    // [[nodiscard]] virtual auto sample_idx(int64_t batch_size) const -> torch::Tensor;
};


#endif 