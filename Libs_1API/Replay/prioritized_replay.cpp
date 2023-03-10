#include "prioritized_replay.h"

#include <cmath>
#include <cstdlib>
#include <random>

#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>

using namespace sycl;


PrioritizedReplayBuffer::PrioritizedReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec,
                                            int64_t batch_size)
        : ReplayBuffer(capacity, data_spec, batch_size) {

}

torch::Tensor PrioritizedReplayBuffer::generate_idx() const {
    auto idx = torch::randint(size(), {m_batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    std::cout<<"Uniform generate idx"<<idx<<".\n";
    return idx;
}

template<typename K, typename V>
void print_map(std::unordered_map<K, V> const &m)
{
    for (auto const &pair: m) {
        std::cout << "{" << pair.first << ": " << pair.second << "}\n";
    }
}

/*
int main() {
    default_selector d_selector;
    SumTreeNary PTree(1024, 16); //size, fanout
    // Test: insert (update) prorities for the first 512=128*4 leaf nodes (no sycl)
    for (int i=0;i<128;i++){
        PTree.set(torch::tensor({i*4,i*4+1,i*4+2,i*4+3}), //data storage indices
        torch::tensor({0.1*i*4,0.1*(i*4+1),0.1*(i*4+2),0.1*(i*4+3)})); //synthetic priority values
    }
    // Test: sampling priorities (yes sycl parallelized, vector size of 8)
    // try {
    // queue q(d_selector, exception_handler);
    queue q(d_selector);
    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
                << q.get_device().get_info<info::device::name>() << "\n";
    // Sampling in dpc++
    IntVector sampled_ind(8); //this output vector size needs to be consistent with the value tensor size passed into prefix_sum function
    PTree.get_prefix_sum_idx_sycl(q, torch::rand(8), sampled_ind);
    // BasicPolicy(q, state_vec, param_vec, a);
    std::cout << "sampled indices from: " << sampled_ind[0] <<" to "<< sampled_ind[7] << "\n";
    // } catch (exception const &e) {
    // std::cout << "An exception is caught for Basic Policy.\n";
    // std::terminate();
    // }
    // auto sampled = PTree.get_prefix_sum_idx(torch::rand(8));

    // std::cout << sampled << std::endl; 
}

*/