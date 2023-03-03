#include "PER.h"

#include <torch/torch.h>
#include <c10/util/ArrayRef.h>



PrioritizedExperienceReplay::PrioritizedExperienceReplay(int64_t size) {
    capacity = size;
}

void PrioritizedExperienceReplay::push(torch::Tensor state,torch::Tensor new_state,torch::Tensor action,torch::Tensor done,torch::Tensor reward, float_t td_error, int64_t ind){
    float_t error(td_error);
    int64_t index(ind);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample (state, new_state, action, reward, done);
    element sample_struct(error, index, sample);

    if (buffer.size() < capacity){
        buffer.push(sample_struct);
    }
    else {
        while (buffer.size() >= capacity) {
            buffer.pop();
        }
        buffer.push(sample_struct);
    }
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
PrioritizedExperienceReplay::sample_queue(int64_t batch_size){
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> b(batch_size);
    while (batch_size > 0 and buffer.size() > 0){
        element s = buffer.top();
        buffer.pop();
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample = s.transition;
        b.push_back(sample);
    }
    return b;
}

int64_t PrioritizedExperienceReplay::size_buffer(){

    return buffer.size();
}

int main(){
    PrioritizedExperienceReplay buffer1(1024);

    // Insertion
    for (int64_t i=0;i<1024;i++){
        torch::Tensor rand_state = torch::rand({1, 4}); 
        torch::Tensor rand_next_state = torch::rand({1, 4}); 
        torch::Tensor rand_action= torch::rand({1}); 
        torch::Tensor rand_done= torch::rand({1}); 
        torch::Tensor rand_reward= torch::rand({1}); 
        float_t td_error = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        buffer1.push(rand_state, rand_next_state, rand_action, rand_done, rand_reward,td_error,i);
    }
   
   int64_t batch_size=8;
    // Sampling
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch =
                buffer1.sample_queue(batch_size);   
    for (size_t i=0;i<batch_size;i++){
        std::cout<<std::get<0>(batch[i])<<" "<<"\n";
        
    }
    
}