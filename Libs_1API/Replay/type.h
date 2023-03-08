#ifndef HIPC21_TYPE_H
#define HIPC21_TYPE_H

#include <torch/torch.h>
#include <string>
#include <unordered_map>


struct MyDataSpec {
    torch::Dtype m_dtype; //For torch::Dtype: kUInt8, kInt8, kInt16, kInt32, kInt64, kFloat32 and kFloat64,
    
    std::vector<int64_t> m_shape;
    
    MyDataSpec(std::vector<int64_t> shape, torch::Dtype dtype = torch::kFloat32) : m_dtype(dtype), m_shape(std::move(shape)) {

    }
};

typedef std::unordered_map<std::string, torch::Tensor> str_to_tensor;
typedef std::unordered_map<std::string, torch::autograd::variable_list> str_to_tensor_list;
typedef std::unordered_map<std::string, MyDataSpec> str_to_dataspec;



#endif //HIPC21_TYPE_H