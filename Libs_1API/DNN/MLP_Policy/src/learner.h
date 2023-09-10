#ifndef LEARNER_H
#define LEARNER_H

#include <torch/torch.h>

#include "dqn.h"

// torch::set_num_threads(1);
// torch::set_num_interop_threads(1);

class Trainer{

    // private: ExperienceReplay buffer;
    private: DQN network, target_network;
    private: torch::optim::Adam dqn_optimizer;
    private: double epsilon_start = 1.0;
    private: double epsilon_final = 0.01;
    private: int64_t epsilon_decay = 30000;
    private: int64_t batch_size = 8;
    private: float gamma = 0.99;

    public:
        Trainer(int64_t input_channels, int64_t num_actions, int64_t capacity);
        torch::Tensor compute_td_loss(int64_t batch_size, float gamma,
                                    torch::Tensor states_tensor,
                                    torch::Tensor new_states_tensor,
                                    torch::Tensor actions_tensor,
                                    torch::Tensor rewards_tensor,
                                    torch::Tensor dones_tensor);
        void load_enviroment(int64_t random_seed, std::string rom_path);
        double epsilon_by_frame(int64_t frame_id);
        torch::Tensor get_tensor_observation(std::vector<unsigned char> state);
        void loadstatedict(torch::nn::Module& model,
                           torch::nn::Module& target_model);
        void trainloop(int64_t random_seed, std::string rom_path, int64_t num_epochs);

};

#endif