#include "sum_tree_nary.h"

#include <omp.h>

template<typename T>
std::vector<T> convert_tensor_to_flat_vector(const torch::Tensor &tensor) {
    torch::Tensor t = torch::flatten(tensor.cpu());
    return {t.data_ptr<T>(), t.data_ptr<T>() + t.numel()};
}

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
    node_idx = get_node_idx_after_padding(node_idx); //node_idx + m_padding;
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

torch::Tensor  SumTreeNary::operator[](const torch::Tensor &idx) const {
    auto idx_vector = convert_tensor_to_flat_vector<int64_t>(idx);
    auto output = torch::zeros_like(idx, torch::TensorOptions().dtype(torch::kFloat32));
    for (int i = 0; i < (int) idx_vector.size(); ++i) {
        output.index_put_({i}, get_value(convert_to_node_idx(idx_vector.at(i))));
    }
    return (output);
}

// Insertion, Update
void  SumTreeNary::set(const torch::Tensor &idx, const torch::Tensor &value) {
    auto idx_vec = convert_tensor_to_flat_vector<int64_t>(idx);
    auto value_vec = convert_tensor_to_flat_vector<float>(value);
    // put all the values
    for (int i = 0; i < (int) idx_vec.size(); ++i) {
        // get data pos
        int64_t pos = idx_vec.operator[](i);
        // get node pos
        pos = convert_to_node_idx(pos); 
        // set the value of the leaf node
        auto original_value = get_value(pos);
        auto new_value = value_vec.operator[](i);
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

void  SumTreeNary::set_sycl(queue &q, const torch::Tensor &idx, const torch::Tensor &value) {
    auto idx_vec = convert_tensor_to_flat_vector<int64_t>(idx);
    auto value_vec = convert_tensor_to_flat_vector<float>(value);

    range<1> num_items{idx_vec.size()};

    range<1> num_items2{(size_t) m_size};

    sycl::buffer a_buf(idx_vec);
    sycl::buffer b_buf(value_vec);

    auto buf_m_bound = sycl::buffer{&m_bound, sycl::range{1}};
    auto buf_m_padding = sycl::buffer{&m_padding, sycl::range{1}};

    // moving complete tree between host and device only for sampling is time-consuming. optimize: manage the tree on device
    std::vector<float> m_values_vec(m_size);
    for (size_t i=0;i<m_size;i++){
        m_values_vec[i]=m_values[i];
    }
    // sycl::buffer buf_m_values(m_values_vec); 
    buffer buf_m_values(m_values_vec.data(), num_items2);
    // auto buf_m_values = sycl::buffer{&m_values_vec, sycl::range{1}};

    auto buf_log2_m_n = sycl::buffer{&log2_m_n, sycl::range{1}};

    q.submit([&](handler &h) {
        accessor ivec(a_buf, h, read_only);
        accessor vvec(b_buf, h, read_only);
        accessor acc_m_bound(buf_m_bound, h, read_only);
        accessor acc_m_padding(buf_m_padding, h, read_only);
        accessor acc_m_values(buf_m_values, h, read_write);
        accessor acc_log2_m_n(buf_log2_m_n, h, read_only);
        h.parallel_for(num_items, [=](auto i) {
            // get data pos
            int64_t pos = ivec[i];
            //convert data index to node position
            pos = pos + acc_m_bound[0]; 
            // set the value of the leaf node
            int64_t pos2 = pos + acc_m_padding[0];
            auto original_value = acc_m_values[pos2];
            auto new_value = vvec[i];
            auto delta = new_value - original_value;
            // update the parent
            while (true) {
                // set_value(pos, get_value(pos) + delta);
                pos2 = pos + acc_m_padding[0];
                auto value_to_set = acc_m_values[pos2] + delta;
                acc_m_values[pos2] = value_to_set;
                // std::cout << "updated entry "<<pos2<<" to be: "<<acc_m_values[pos2];
                if (pos == 0) { //reached root
                    break;
                }
                // pos = get_parent(pos);
                pos= (pos - 1) >> acc_log2_m_n[0];
            }
        });
    });

    // Wait until compute tasks on GPU done
    q.wait();
    // for (size_t i=0;i<m_size;i++){
    //     // m_values[i]=buf_m_values[i];
    //     m_values[i]=m_values_vec[i];
    // }

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
torch::Tensor  SumTreeNary::get_prefix_sum_idx(torch::Tensor value) const {
    auto value_vec = convert_tensor_to_flat_vector<float>(value);
    auto index = torch::ones_like(value, torch::TensorOptions().dtype(torch::kInt64));

    for (int i = 0; i < (int) value_vec.size(); i++) {
        int64_t idx = get_root();
        float current_val = value_vec[i];
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
        index.index_put_({i}, convert_to_data_idx(idx));
    }

    return index;
}



//************************************
// Sampling in SYCL on device
//************************************
void SumTreeNary::get_prefix_sum_idx_sycl(queue &q, torch::Tensor value, IntVector &index_parallel) {
    auto value_vec = convert_tensor_to_flat_vector<float>(value);
    // Create the range object for the vectors managed by the buffer.
    range<1> num_items{value_vec.size()};

    // Create buffers that hold the data shared between the host and the devices.
    buffer a_buf(value_vec);
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
        
        h.parallel_for(num_items, [=](auto i) {
            int64_t idx = 0;
            float current_val = vvec[i];
            while (idx<acc_m_bound[0]) { //!is_leaf(idx)
                idx = (idx << acc_log2_m_n[0]) + 1; //get_left_child(idx);
                float partial_sum = 0.;
                for (int64_t j = 0; j < acc_m_n[0]; ++j) {
                    //get node idx after padding
                    idx = idx + acc_m_padding[0]; 
                    //get traversed priority value sum at idx
                    float after_sum = acc_m_values[idx] + partial_sum; 
                    if (after_sum >= current_val) {
                        break; //target priority value reached, sample the current idx
                    }
                    // target priority value not reached, get next sibling
                    partial_sum = after_sum;
                    idx += 1;
                }
                current_val -= partial_sum;
            } 
            out_index[i]= idx-acc_m_bound[0]; //convert_to_data_storage_idx(idx);
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

int main() {
    int batchsize_sampling=16;
    int batchsize_update=16;
    default_selector d_selector;
    queue q(d_selector);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
                << q.get_device().get_info<info::device::name>() << "\n";

    SumTreeNary PTree(1024*8, 16); //size, fanout
    // Test: insert (update) prorities for the first 512=128*4 leaf nodes (sycl)
    double tstart = omp_get_wtime();
    // torch::Tensor ds_indices=torch::empty(batchsize_update);
    // torch::Tensor ds_values=torch::empty(batchsize_update);

    for (int i=0;i<2;i++){
        // for (int j=0; j<batchsize_update; j++){
        //     ds_indices.index_put_({j},i*batchsize_update+j);
        //     ds_values.index_put_({j},0.1*(i*batchsize_update+j));
        // }
        // PTree.set(ds_indices, //data storage indices
        // ds_values); //synthetic priority values
        // PTree.set_sycl(q, ds_indices,ds_values); 
        
        ////bs 128
        // PTree.set( torch::tensor({i*128+0,i*128+1,i*128+2,i*128+3,i*128+4,i*128+5,i*128+6,i*128+7,i*128+8,i*128+9,i*128+10,i*128+11,i*128+12,i*128+13,i*128+14,i*128+15,i*128+16,i*128+17,i*128+18,i*128+19,i*128+20,i*128+21,i*128+22,i*128+23,i*128+24,i*128+25,i*128+26,i*128+27,i*128+28,i*128+29,i*128+30,i*128+31,i*128+32,i*128+33,i*128+34,i*128+35,i*128+36,i*128+37,i*128+38,i*128+39,i*128+40,i*128+41,i*128+42,i*128+43,i*128+44,i*128+45,i*128+46,i*128+47,i*128+48,i*128+49,i*128+50,i*128+51,i*128+52,i*128+53,i*128+54,i*128+55,i*128+56,i*128+57,i*128+58,i*128+59,i*128+60,i*128+61,i*128+62,i*128+63,i*128+64,i*128+65,i*128+66,i*128+67,i*128+68,i*128+69,i*128+70,i*128+71,i*128+72,i*128+73,i*128+74,i*128+75,i*128+76,i*128+77,i*128+78,i*128+79,i*128+80,i*128+81,i*128+82,i*128+83,i*128+84,i*128+85,i*128+86,i*128+87,i*128+88,i*128+89,i*128+90,i*128+91,i*128+92,i*128+93,i*128+94,i*128+95,i*128+96,i*128+97,i*128+98,i*128+99,i*128+100,i*128+101,i*128+102,i*128+103,i*128+104,i*128+105,i*128+106,i*128+107,i*128+108,i*128+109,i*128+110,i*128+111,i*128+112,i*128+113,i*128+114,i*128+115,i*128+116,i*128+117,i*128+118,i*128+119,i*128+120,i*128+121,i*128+122,i*128+123,i*128+124,i*128+125,i*128+126,i*128+127}),
        // torch::tensor({0.1*(i*128+0),0.1*(i*128+1),0.1*(i*128+2),0.1*(i*128+3),0.1*(i*128+4),0.1*(i*128+5),0.1*(i*128+6),0.1*(i*128+7),0.1*(i*128+8),0.1*(i*128+9),0.1*(i*128+10),0.1*(i*128+11),0.1*(i*128+12),0.1*(i*128+13),0.1*(i*128+14),0.1*(i*128+15),0.1*(i*128+16),0.1*(i*128+17),0.1*(i*128+18),0.1*(i*128+19),0.1*(i*128+20),0.1*(i*128+21),0.1*(i*128+22),0.1*(i*128+23),0.1*(i*128+24),0.1*(i*128+25),0.1*(i*128+26),0.1*(i*128+27),0.1*(i*128+28),0.1*(i*128+29),0.1*(i*128+30),0.1*(i*128+31),0.1*(i*128+32),0.1*(i*128+33),0.1*(i*128+34),0.1*(i*128+35),0.1*(i*128+36),0.1*(i*128+37),0.1*(i*128+38),0.1*(i*128+39),0.1*(i*128+40),0.1*(i*128+41),0.1*(i*128+42),0.1*(i*128+43),0.1*(i*128+44),0.1*(i*128+45),0.1*(i*128+46),0.1*(i*128+47),0.1*(i*128+48),0.1*(i*128+49),0.1*(i*128+50),0.1*(i*128+51),0.1*(i*128+52),0.1*(i*128+53),0.1*(i*128+54),0.1*(i*128+55),0.1*(i*128+56),0.1*(i*128+57),0.1*(i*128+58),0.1*(i*128+59),0.1*(i*128+60),0.1*(i*128+61),0.1*(i*128+62),0.1*(i*128+63),0.1*(i*128+64),0.1*(i*128+65),0.1*(i*128+66),0.1*(i*128+67),0.1*(i*128+68),0.1*(i*128+69),0.1*(i*128+70),0.1*(i*128+71),0.1*(i*128+72),0.1*(i*128+73),0.1*(i*128+74),0.1*(i*128+75),0.1*(i*128+76),0.1*(i*128+77),0.1*(i*128+78),0.1*(i*128+79),0.1*(i*128+80),0.1*(i*128+81),0.1*(i*128+82),0.1*(i*128+83),0.1*(i*128+84),0.1*(i*128+85),0.1*(i*128+86),0.1*(i*128+87),0.1*(i*128+88),0.1*(i*128+89),0.1*(i*128+90),0.1*(i*128+91),0.1*(i*128+92),0.1*(i*128+93),0.1*(i*128+94),0.1*(i*128+95),0.1*(i*128+96),0.1*(i*128+97),0.1*(i*128+98),0.1*(i*128+99),0.1*(i*128+100),0.1*(i*128+101),0.1*(i*128+102),0.1*(i*128+103),0.1*(i*128+104),0.1*(i*128+105),0.1*(i*128+106),0.1*(i*128+107),0.1*(i*128+108),0.1*(i*128+109),0.1*(i*128+110),0.1*(i*128+111),0.1*(i*128+112),0.1*(i*128+113),0.1*(i*128+114),0.1*(i*128+115),0.1*(i*128+116),0.1*(i*128+117),0.1*(i*128+118),0.1*(i*128+119),0.1*(i*128+120),0.1*(i*128+121),0.1*(i*128+122),0.1*(i*128+123),0.1*(i*128+124),0.1*(i*128+125),0.1*(i*128+126),0.1*(i*128+127)})); //synthetic priority values
        
        ////bs 64
        // PTree.set( torch::tensor({i*64+0,i*64+1,i*64+2,i*64+3,i*64+4,i*64+5,i*64+6,i*64+7,i*64+8,i*64+9,i*64+10,i*64+11,i*64+12,i*64+13,i*64+14,i*64+15,i*64+16,i*64+17,i*64+18,i*64+19,i*64+20,i*64+21,i*64+22,i*64+23,i*64+24,i*64+25,i*64+26,i*64+27,i*64+28,i*64+29,i*64+30,i*64+31,i*64+32,i*64+33,i*64+34,i*64+35,i*64+36,i*64+37,i*64+38,i*64+39,i*64+40,i*64+41,i*64+42,i*64+43,i*64+44,i*64+45,i*64+46,i*64+47,i*64+48,i*64+49,i*64+50,i*64+51,i*64+52,i*64+53,i*64+54,i*64+55,i*64+56,i*64+57,i*64+58,i*64+59,i*64+60,i*64+61,i*64+62,i*64+63}),
        // torch::tensor({0.1*(i*64+0),0.1*(i*64+1),0.1*(i*64+2),0.1*(i*64+3),0.1*(i*64+4),0.1*(i*64+5),0.1*(i*64+6),0.1*(i*64+7),0.1*(i*64+8),0.1*(i*64+9),0.1*(i*64+10),0.1*(i*64+11),0.1*(i*64+12),0.1*(i*64+13),0.1*(i*64+14),0.1*(i*64+15),0.1*(i*64+16),0.1*(i*64+17),0.1*(i*64+18),0.1*(i*64+19),0.1*(i*64+20),0.1*(i*64+21),0.1*(i*64+22),0.1*(i*64+23),0.1*(i*64+24),0.1*(i*64+25),0.1*(i*64+26),0.1*(i*64+27),0.1*(i*64+28),0.1*(i*64+29),0.1*(i*64+30),0.1*(i*64+31),0.1*(i*64+32),0.1*(i*64+33),0.1*(i*64+34),0.1*(i*64+35),0.1*(i*64+36),0.1*(i*64+37),0.1*(i*64+38),0.1*(i*64+39),0.1*(i*64+40),0.1*(i*64+41),0.1*(i*64+42),0.1*(i*64+43),0.1*(i*64+44),0.1*(i*64+45),0.1*(i*64+46),0.1*(i*64+47),0.1*(i*64+48),0.1*(i*64+49),0.1*(i*64+50),0.1*(i*64+51),0.1*(i*64+52),0.1*(i*64+53),0.1*(i*64+54),0.1*(i*64+55),0.1*(i*64+56),0.1*(i*64+57),0.1*(i*64+58),0.1*(i*64+59),0.1*(i*64+60),0.1*(i*64+61),0.1*(i*64+62),0.1*(i*64+63)}));

        ////bs 32
        // PTree.set(torch::tensor({i*32+0,i*32+1,i*32+2,i*32+3,i*32+4,i*32+5,i*32+6,i*32+7,i*32+8,i*32+9,i*32+10,i*32+11,i*32+12,i*32+13,i*32+14,i*32+15,i*32+16,i*32+17,i*32+18,i*32+19,i*32+20,i*32+21,i*32+22,i*32+23,i*32+24,i*32+25,i*32+26,i*32+27,i*32+28,i*32+29,i*32+30,i*32+31}), //data storage indices
        // torch::tensor({0.1*(i*32+0),0.1*(i*32+1),0.1*(i*32+2),0.1*(i*32+3),0.1*(i*32+4),0.1*(i*32+5),0.1*(i*32+6),0.1*(i*32+7),0.1*(i*32+8),0.1*(i*32+9),0.1*(i*32+10),0.1*(i*32+11),0.1*(i*32+12),0.1*(i*32+13),0.1*(i*32+14),0.1*(i*32+15),0.1*(i*32+16),0.1*(i*32+17),0.1*(i*32+18),0.1*(i*32+19),0.1*(i*32+20),0.1*(i*32+21),0.1*(i*32+22),0.1*(i*32+23),0.1*(i*32+24),0.1*(i*32+25),0.1*(i*32+26),0.1*(i*32+27),0.1*(i*32+28),0.1*(i*32+29),0.1*(i*32+30),0.1*(i*32+31)})); //synthetic priority values

        ////bs 16
        // PTree.set(torch::tensor({i*16+0,i*16+1,i*16+2,i*16+3,i*16+4,i*16+5,i*16+6,i*16+7,i*16+8,i*16+9,i*16+10,i*16+11,i*16+12,i*16+13,i*16+14,i*16+15}),
        // torch::tensor({0.1*(i*16+0),0.1*(i*16+1),0.1*(i*16+2),0.1*(i*16+3),0.1*(i*16+4),0.1*(i*16+5),0.1*(i*16+6),0.1*(i*16+7),0.1*(i*16+8),0.1*(i*16+9),0.1*(i*16+10),0.1*(i*16+11),0.1*(i*16+12),0.1*(i*16+13),0.1*(i*16+14),0.1*(i*16+15)}));
        
        ////bs 8
        PTree.set( torch::tensor({i*8+0,i*8+1,i*8+2,i*8+3,i*8+4,i*8+5,i*8+6,i*8+7}),
        torch::tensor({0.1*(i*8+0),0.1*(i*8+1),0.1*(i*8+2),0.1*(i*8+3),0.1*(i*8+4),0.1*(i*8+5),0.1*(i*8+6),0.1*(i*8+7)}));
    
    }

    double tstop = omp_get_wtime();

    // std::cout<<"Updated priority values in the first 16 entries: ";
    // std::cout<<PTree.operator[](torch::tensor({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}))<<"\n";

    printf("Update of batch %d took %f seconds per iteration.\n", batchsize_update, (tstop - tstart)/2);
    // Sampling in dpc++
    IntVector sampled_ind(batchsize_sampling); //this output vector size needs to be consistent with the value tensor size passed into prefix_sum function
    tstart = omp_get_wtime();
    
    for (int i=0;i<2;i++){
        torch::Tensor values=torch::rand(batchsize_sampling)*batchsize_sampling;
        PTree.get_prefix_sum_idx_sycl(q, values, sampled_ind);
        // std::cout << "emulated priority values: " <<values;
    }
    
    tstop = omp_get_wtime();

    // std::cout << "sampled indices: ";
    // for (int i=0; i<batchsize_sampling; i++){
    //     std::cout<< sampled_ind[i] <<" ";
    // } 
    std::cout<< "\n";
    printf("Sampling of batch %d took %f seconds.\n", batchsize_sampling, (tstop - tstart)/2);

}