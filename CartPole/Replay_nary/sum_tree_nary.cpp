#include "sum_tree_nary.h"


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

float  SumTreeNary::reduce() const {
    return get_value(get_root());
}

float  SumTreeNary::reduce(int64_t start, int64_t end) const {
    assert(start >= 0 && end <= size() && end >= start);
    if (start == 0) {
        return reduce(end);
    } else return reduce(end) - reduce(start);
}

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

// Sampling
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