#ifndef PRIORITIZED_REPLAY_H
#define PRIORITIZED_REPLAY_H

#include "replay_buffer_base.h"
#include "sum_tree_nary.h"


class PrioritizedReplayBuffer final : public ReplayBuffer {
public:
    explicit PrioritizedReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec, int64_t batch_size);

    [[nodiscard]] torch::Tensor generate_idx() const override;

};

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

template<typename K, typename V>
void print_map(std::unordered_map<K, V> const &m);

#endif //HIPC21_UNIFORM_REPLAY_BUFFER_H