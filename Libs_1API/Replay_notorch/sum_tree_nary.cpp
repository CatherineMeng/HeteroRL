#include "sum_tree_nary.h"



// template<typename T>
// std::vector<T> convert_tensor_to_flat_vector(const torch::Tensor &tensor) {
//     torch::Tensor t = torch::flatten(tensor.cpu());
//     return {t.data_ptr<T>(), t.data_ptr<T>() + t.numel()};
// }
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

SumTreeNary::SumTreeNary(int64_t size, int64_t n) :
        m_n(n), //fanout
        m_size(size) { //leaf level size
    last_level_size = 1;
    while (last_level_size < size) {
        last_level_size = last_level_size * m_n;
    }
    m_bound = (last_level_size - 1) / (m_n - 1);
    initialize();
}

int64_t  SumTreeNary::size() const {
    return m_size;
}

int64_t  SumTreeNary::get_node_idx_after_padding(int64_t node_idx) const {
    return node_idx + m_padding;
}

float  SumTreeNary::get_value(int64_t node_idx) const {
    node_idx = get_node_idx_after_padding(node_idx);
    auto value = m_values[node_idx];
    return value;
}

void  SumTreeNary::set_value(int64_t node_idx, float value) {
    node_idx = get_node_idx_after_padding(node_idx);
    m_values[node_idx] = value;
}

int64_t  SumTreeNary::convert_to_node_idx(int64_t data_idx) const {
    return data_idx + m_bound;
}

int64_t  SumTreeNary::convert_to_data_idx(int64_t node_idx) const {
    return node_idx - m_bound;
}

int64_t  SumTreeNary::get_parent(int64_t node_idx) const {
    return (node_idx - 1) >> log2_m_n;
}

int64_t  SumTreeNary::get_root() const {
    return 0;
}

// torch::Tensor  SumTreeNary::operator[](const torch::Tensor &idx) const {
//     auto idx_vector = convert_tensor_to_flat_vector<int64_t>(idx);
//     auto output = torch::zeros_like(idx, torch::TensorOptions().dtype(torch::kFloat32));
//     for (int i = 0; i < (int) idx_vector.size(); ++i) {
//         output.index_put_({i}, get_value(convert_to_node_idx(idx_vector.at(i))));
//     }
//     return (output);
// }

std::vector<float>  SumTreeNary::operator[](const std::vector<int64_t> &idx) const {
    std::vector<float> output(idx.size());
    for (int i = 0; i < (int) idx.size(); ++i) {
        output[i]=get_value(idx[i]);
    }
    return (output);
}

// Insertion, Update
// void  SumTreeNary::set(const torch::Tensor &idx, const torch::Tensor &value) {
//     auto idx_vec = convert_tensor_to_flat_vector<int64_t>(idx);
//     auto value_vec = convert_tensor_to_flat_vector<float>(value);
//     // put all the values
//     for (int i = 0; i < (int) idx_vec.size(); ++i) {
//         // get data pos
//         int64_t pos = idx_vec.operator[](i);
//         // get node pos
//         pos = convert_to_node_idx(pos);
//         // set the value of the leaf node
//         auto original_value = get_value(pos);
//         auto new_value = value_vec.operator[](i);
//         auto delta = new_value - original_value;
//         // update the parent
//         while (true) {
//             set_value(pos, get_value(pos) + delta);
//             if (pos == get_root()) {
//                 break;
//             }
//             pos = get_parent(pos);
//         }
//     }
// }

void  SumTreeNary::set(const std::vector<int64_t> &idx, const std::vector<float> &value) { 
    //assumes input is already flat vactor
    // put all the values
    for (int i = 0; i < (int) idx.size(); ++i) {
        // get data pos
        int64_t pos = idx[i];
        // get node pos
        pos = convert_to_node_idx(pos);
        // set the value of the leaf node
        auto original_value = get_value(pos);
        auto new_value = value[i];
        auto delta = new_value - original_value;
        // update the parent
        while (true) {
            set_value(pos, get_value(pos) + delta);
            if (pos == get_root()) {
                break;
            }
            pos = get_parent(pos);
        }
    }
}


float  SumTreeNary::reduce() const {
    return get_value(get_root());
}

float  SumTreeNary::reduce(int64_t start, int64_t end) const {
    assert(start >= 0 && end <= size() && end >= start);
    if (start == 0) {
        return reduce(end);
    } else return reduce(end) - reduce(start);
}
// obtain prefix sum value from first element to element indexed at end. (index in the data storage)
float  SumTreeNary::reduce(int64_t end) const {
    assert(end > 0 && end <= size());
    if (end == size()) {
        return reduce();
    }
    end = convert_to_node_idx(end);
    float result = 0.;
    while (end != get_root()) {
        // sum all the node left to it.
        int64_t parent = get_parent(end);
        int64_t left_child = get_left_child(parent);
        while (true) {
            if (left_child != end) {
                result += get_value(left_child);
            } else {
                break;
            }
            left_child += 1;
        }
        end = parent;
    }
    return result;
}

// Sampling. 
// Inout: random generated value that is the target prefix-sum. 
// Output: the index used to access the data storage - the sampled exp whose priority sums up to value according to the current priority distribution.
// torch::Tensor  SumTreeNary::get_prefix_sum_idx(torch::Tensor value) const {
//     auto value_vec = convert_tensor_to_flat_vector<float>(value);
//     auto index = torch::ones_like(value, torch::TensorOptions().dtype(torch::kInt64));

//     for (int i = 0; i < (int) value_vec.size(); i++) {
//         int64_t idx = get_root();
//         float current_val = value_vec[i];
//         while (!is_leaf(idx)) {
//             idx = get_left_child(idx);
//             float partial_sum = 0.;
//             for (int64_t j = 0; j < m_n; ++j) {
//                 float after_sum = get_value(idx) + partial_sum;
//                 if (after_sum >= current_val) {
//                     break;
//                 }
//                 // get next sibling
//                 partial_sum = after_sum;
//                 idx += 1;
//             }
//             current_val -= partial_sum;
//         }
//         index.index_put_({i}, convert_to_data_idx(idx));
//     }

//     return index;
// }

std::vector<int64_t>  SumTreeNary::get_prefix_sum_idx(const std::vector<float> &value) const {

    std::vector<int64_t> index(value.size());

    for (int i = 0; i < (int) value.size(); i++) {
        int64_t idx = get_root();
        float current_val = value[i];
        while (!is_leaf(idx)) {
            idx = get_left_child(idx);
            float partial_sum = 0.;
            for (int64_t j = 0; j < m_n; ++j) {
                float after_sum = get_value(idx) + partial_sum;
                if (after_sum >= current_val) {
                    break;
                }
                // get next sibling
                partial_sum = after_sum;
                idx += 1;
            }
            current_val -= partial_sum;
        }
        index[i]=convert_to_data_idx(idx);
    }

    return index;
}



//************************************
// Sampling in SYCL on device
//************************************
// void SumTreeNary::get_prefix_sum_idx_sycl(queue &q, torch::Tensor value, IntVector &index_parallel) {
//     auto value_vec = convert_tensor_to_flat_vector<float>(value);
//     // Create the range object for the vectors managed by the buffer.
//     range<1> num_items{value_vec.size()};

//     // Create buffers that hold the data shared between the host and the devices.
//     buffer a_buf(value_vec);
//     buffer out_buf(index_parallel.data(), num_items);
//     auto buf_m_bound = sycl::buffer{&m_bound, sycl::range{1}};
//     auto buf_m_n = sycl::buffer{&m_n, sycl::range{1}};
//     auto buf_log2_m_n = sycl::buffer{&log2_m_n, sycl::range{1}};
//     auto buf_m_padding = sycl::buffer{&m_padding, sycl::range{1}};
//     // moving complete tree between host and device only for sampling is time-consuming. optimize: manage the tree on device
//     std::vector<float> m_values_vec(m_size);
//     for (size_t i=0;i<m_size;i++){
//         m_values_vec[i]=m_values[i];
//     }
//     buffer buf_m_values(m_values_vec); 
//     // Submit a command group to the queue by a lambda function that contains the
//     // data access permission and device computation (kernel).

    
//     q.submit([&](handler &h) {
//         // Create an accessor for each buffer with access permission: read, write or
//         // read/write. The accessor is a mean to access the memory in the buffer.
//         accessor vvec(a_buf, h, read_only);
//         // The sum_accessor is used to store (with write permission) the sum data.
//         accessor out_index(out_buf, h, write_only, no_init);

//         accessor acc_m_bound(buf_m_bound, h, read_only);
//         accessor acc_m_n(buf_m_n, h, read_only);
//         accessor acc_log2_m_n(buf_log2_m_n, h, read_only);
//         accessor acc_m_padding(buf_m_padding, h, read_only);
//         accessor acc_m_values(buf_m_values, h, read_only);

//         // Use parallel_for to run batched sampling in parallel on device.
//         // h.parallel_for(num_items, [=](auto i) {
        
//         h.parallel_for(num_items, [=](auto i) {
//             int64_t idx = 0;
            
//             float current_val = vvec[i];
//             while (idx<acc_m_bound[0]) { //!is_leaf(idx)
//                 idx = (idx << acc_log2_m_n[0]) + 1; //get_left_child(idx);
//                 float partial_sum = 0.;
//                 for (int64_t j = 0; j < acc_m_n[0]; ++j) {
//                     // float after_sum = get_value(idx) + partial_sum;
//                     idx = idx + acc_m_padding[0]; //get_node_idx_after_padding(idx); for get_value(idx)
//                     float after_sum = acc_m_values[idx] + partial_sum; //get_value(idx)
                    
//                     if (after_sum >= current_val) {
//                         break;
//                     }
//                     // get next sibling
//                     partial_sum = after_sum;
//                     idx += 1;
//                 }
//                 current_val -= partial_sum;
//             }
            
//             out_index[i]= idx-acc_m_bound[0]; //convert_to_data_idx(idx);
//             // index.index_put_({i}, convert_to_data_idx(idx));
    
//         });
        
//     });

//     // Wait until compute tasks on GPU done
//     q.wait();
    
// }
void SumTreeNary::get_prefix_sum_idx_sycl(queue &q, std::vector<float> value, IntVector &index_parallel) {
    // Create the range object for the vectors managed by the buffer.
    range<1> num_items{value.size()};

    // Create buffers that hold the data shared between the host and the devices.
    buffer a_buf(value);
    buffer out_buf(index_parallel.data(), num_items);
    auto buf_m_bound = sycl::buffer{&m_bound, sycl::range{1}};
    auto buf_m_n = sycl::buffer{&m_n, sycl::range{1}};
    auto buf_log2_m_n = sycl::buffer{&log2_m_n, sycl::range{1}};
    auto buf_m_padding = sycl::buffer{&m_padding, sycl::range{1}};
    // moving complete tree between host and device only for sampling is time-consuming. optimize: manage the tree on device
    std::vector<float> m_values_vec(m_size);
    for (size_t i=0;i<m_size;i++){
        m_values_vec[i]=m_values[i];
    }
    buffer buf_m_values(m_values_vec); 
    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    std::cout << "Created buffers "<< "\n";
    
    q.submit([&](handler &h) {
        // Create an accessor for each buffer with access permission: read, write or
        // read/write. The accessor is a mean to access the memory in the buffer.
        accessor vvec(a_buf, h, read_only);
        // The sum_accessor is used to store (with write permission) the sum data.
        accessor out_index(out_buf, h, write_only, no_init);

        accessor acc_m_bound(buf_m_bound, h, read_only);
        accessor acc_m_n(buf_m_n, h, read_only);
        accessor acc_log2_m_n(buf_log2_m_n, h, read_only);
        accessor acc_m_padding(buf_m_padding, h, read_only);
        accessor acc_m_values(buf_m_values, h, read_only);

        // Use parallel_for to run batched sampling in parallel on device.
        // h.parallel_for(num_items, [=](auto i) {
        
        h.parallel_for(num_items, [=](auto i) {
            int64_t idx = 0;
            
            float current_val = vvec[i];
            while (idx<acc_m_bound[0]) { //!is_leaf(idx)
                idx = (idx << acc_log2_m_n[0]) + 1; //get_left_child(idx);
                float partial_sum = 0.;
                for (int64_t j = 0; j < acc_m_n[0]; ++j) {
                    // float after_sum = get_value(idx) + partial_sum;
                    idx = idx + acc_m_padding[0]; //get_node_idx_after_padding(idx); for get_value(idx)
                    float after_sum = acc_m_values[idx] + partial_sum; //get_value(idx)
                    
                    if (after_sum >= current_val) {
                        break;
                    }
                    // get next sibling
                    partial_sum = after_sum;
                    idx += 1;
                }
                current_val -= partial_sum;
            }
            
            out_index[i]= idx-acc_m_bound[0]; //convert_to_data_idx(idx);
            // index.index_put_({i}, convert_to_data_idx(idx));
    
        });
        
    });

    // Wait until compute tasks on GPU done
    q.wait();
    
}

bool  SumTreeNary::is_leaf(int64_t node_idx) const {
    return node_idx >= m_bound;
}

int64_t  SumTreeNary::get_left_child(int64_t node_idx) const {
    // using shift operator is crucial
    return (node_idx << log2_m_n) + 1;
}

void  SumTreeNary::initialize() {
    // zero-based indexing
    int64_t total_size = (last_level_size * m_n - 1) / (m_n - 1);
    // making the data at each level cache aligned
    m_padding = m_n - 1;
    log2_m_n = (int64_t) std::log2(m_n);
    m_values = new float[total_size + m_padding];
    for (int i = 0; i < total_size; ++i) {
        m_values[i] = 0.;
    }
//    spdlog::info("SumTreeNary, n = {0}, size = {1}, m_bound = {2}", m_n, m_size, m_bound);
}

/*
int main(){
    SumTreeNary PTree(1024, 16);
    // Test: insert (update) prorities for the first 512=128*4 leaf nodes
    for (int i=0;i<128;i++){
        PTree.set(torch::tensor({i*4,i*4+1,i*4+2,i*4+3}),
        torch::tensor({0.1*i*4,0.1*(i*4+1),0.1*(i*4+2),0.1*(i*4+3)}));
    }
    // Test: sampling priorities

    auto sampled = PTree.get_prefix_sum_idx(torch::rand(4));
    std::cout << sampled << std::endl; 

}
*/
int main() {
    default_selector d_selector;
    SumTreeNary PTree(1024, 16); //size, fanout
    // Test: insert (update) prorities for the first 512=128*4 leaf nodes (no sycl)
    for (int i=0;i<128;i++){
        std::vector<int64_t> idx_vect{(int64_t)i*4,(int64_t)i*4+1,(int64_t)i*4+2,(int64_t)i*4+3};
        std::vector<float> val_vect{(float)0.1*i*4,(float)0.1*(i*4+1),(float)0.1*(i*4+2),(float)0.1*(i*4+3)};
        // PTree.set(torch::tensor({i*4,i*4+1,i*4+2,i*4+3}), //data storage indices
        // torch::tensor({0.1*i*4,0.1*(i*4+1),0.1*(i*4+2),0.1*(i*4+3)})); //synthetic priority values
        PTree.set(idx_vect, //data storage indices
        val_vect); //synthetic priority values
    }
    // Test: sampling priorities (yes sycl parallelized, vector size of 8)
    try {
    queue q(d_selector, exception_handler);
    // queue q(d_selector);
    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
                << q.get_device().get_info<info::device::name>() << "\n";
    // Sampling in dpc++
    IntVector sampled_ind(8); //this output vector size needs to be consistent with the value tensor size passed into prefix_sum function
    std::vector<float> sampling_values(8);
    // float r2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X)); //generate a random float from 0 to X
    for (size_t i=0;i<8;i++){
        // float r = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/5));
        sampling_values[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/65));
    }
    std::cout << "Created emulated sampling values: "<<sampling_values[0]<<sampling_values[7]<< "\n";
    PTree.get_prefix_sum_idx_sycl(q, sampling_values, sampled_ind);
    // BasicPolicy(q, state_vec, param_vec, a);
    std::cout << "sampled indices: " << sampled_ind[0] <<" "
    <<sampled_ind[1]<< " "<< sampled_ind[2] <<" "<< sampled_ind[3] <<" "
    << sampled_ind[4] <<" "<< sampled_ind[5] <<" "
    << sampled_ind[6] <<" "<< sampled_ind[7] <<" "<< "\n";
    } catch (exception const &e) {
    std::cout << "An exception is caught for Basic Policy.\n";
    std::terminate();
    }
    // auto sampled = PTree.get_prefix_sum_idx(torch::rand(8));

    // std::cout << sampled << std::endl; 
}