#include "uniform_replay.h"



UniformReplayBuffer::UniformReplayBuffer(int64_t capacity, const str_to_dataspec &data_spec,
                                            int64_t batch_size)
        : ReplayBuffer(capacity, data_spec, batch_size) {

}

torch::Tensor UniformReplayBuffer::generate_idx() const {
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

int main(){
    std::vector<int64_t> vect{ 1024 }; //every experience is vect of 1024
    std::cout<<"step 0"<<"\n";
    // MyDataSpec dspec(vect,torch::kInt32);
    MyDataSpec dspec(vect);
    std::cout<<"step 1"<<"\n";

    str_to_dataspec new_dspecs;
    std::cout<<"step 2"<<"\n";
    // typedef std::unordered_map<std::string, MyDataSpec> str_to_dataspec;
    new_dspecs.insert ( {"replay1",dspec});  
    std::cout<<"step 3"<<"\n";  
    UniformReplayBuffer RBuffer(1024, new_dspecs,8); //1024 experiences in total for m_storage[name]
    std::cout<<"step 4"<<"\n";
    
    // for (int i=0;i<1024/8;i++){
    //     std::cout<<"..."<<i<<"..."<<"\n";
    //     str_to_tensor data1;
    //     data1[std::to_string(i)]=torch::rand({8,16}); //8 batches, each feature size 16 (encode/decode based on env)
    //     RBuffer.add_batch(data1);
    // }
    str_to_tensor data1;
    std::cout<<"step 5"<<"\n";
    data1.insert ( {"replay1",torch::rand({8,1024})});  
    //inserted data - dim0:batch size. dim1:expereince size, must match defined in constructing replay buffer
    std::cout<<"step 6"<<"\n";
    RBuffer.add_batch(data1);
    std::cout<<"step 7"<<"\n";
    
    str_to_tensor data2 = RBuffer.sample(0);
    std::cout<<"step 8"<<"\n";
    // print_map(data2);
}