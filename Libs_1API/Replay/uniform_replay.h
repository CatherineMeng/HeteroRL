#ifndef UNIFORM_REPLAY_H
#define UNIFORM_REPLAY_H

#include "replay_buffer_base.h"


class UniformReplayBuffer final : public ReplayBuffer {
public:
    explicit UniformReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec, int64_t batch_size);

    [[nodiscard]] torch::Tensor generate_idx() const override;


};

template<typename K, typename V>
void print_map(std::unordered_map<K, V> const &m);

#endif //HIPC21_UNIFORM_REPLAY_BUFFER_H