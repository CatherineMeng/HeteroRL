#include "learner.h"
#include "dqn.h"
#include <math.h>
#include <chrono>
typedef std::vector<int> IntVector; 
typedef std::vector<float> FloatVector; 

Trainer::Trainer(int64_t state_dim, int64_t num_actions, int64_t capacity):
        // buffer(capacity), //used for replay, integrate later
        network(state_dim, num_actions),
        target_network(state_dim, num_actions),
        dqn_optimizer(
            // network.parameters(), torch::optim::AdamOptions(0.0001).beta1(0.5)){}
            network.parameters(), torch::optim::AdamOptions(0.0001)){}

    // torch::Tensor Trainer::compute_td_loss(int64_t batch_size, float gamma,
    //     torch::Tensor states_tensor,
    //     torch::Tensor new_states_tensor,
    //     torch::Tensor actions_tensor,
    //     torch::Tensor rewards_tensor,
    //     torch::Tensor dones_tensor){
        
        
    //     torch::Tensor q_values = network.forward(states_tensor);
    //     torch::Tensor next_target_q_values = target_network.forward(new_states_tensor);
    //     torch::Tensor next_q_values = network.forward(new_states_tensor);


    //     actions_tensor = actions_tensor.to(torch::kInt64);

    //     // std::cout << "actions_tensor.unsqueeze(1) dims: " <<  actions_tensor.unsqueeze(1).sizes()<< std::endl;
    //     // std::cout << "q_values.gather(1, actions_tensor.unsqueeze(1)) dims: " <<  q_values.gather(1, actions_tensor.unsqueeze(1)).sizes()<< std::endl;
        
    //     torch::Tensor q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1);
    //     // torch::Tensor q_value = q_values.gather(1, actions_tensor).squeeze(1);
    //     torch::Tensor maximum = std::get<1>(next_q_values.max(1));
    //     torch::Tensor next_q_value = next_target_q_values.gather(1, maximum.unsqueeze(1)).squeeze(1);
    //     torch::Tensor expected_q_value = rewards_tensor + gamma*next_q_value*(1-dones_tensor);
    //     // torch::Tensor loss = torch::mse_loss(q_value, expected_q_value);
    //     // std::cout << "q_value dims: " <<  q_value.sizes()<< std::endl;
    //     // std::cout << "expected_q_value dims: " <<  actions_tensor.sizes()<< std::endl;
    //     // q_value.requires_grad_(true);
    //     // expected_q_value.requires_grad_(true);

    //     torch::Tensor loss = torch::mse_loss(q_value, expected_q_value);
    //     // loss.requires_grad_(true);
    //     // torch::requires_grad()
        

    //     dqn_optimizer.zero_grad();
    //     loss.backward();
    //     dqn_optimizer.step();

    //     return loss;

    // }

    torch::Tensor Trainer::compute_td_loss(int64_t batch_size, float gamma,
        torch::Tensor states_tensor,
        torch::Tensor new_states_tensor,
        torch::Tensor actions_tensor,
        torch::Tensor rewards_tensor,
        torch::Tensor dones_tensor) {

        // Check if GPU is available
        bool use_cuda = torch::cuda::is_available();
        if(use_cuda){
            std::cout<<"Using GPU!\n";
        }
        torch::Device device = use_cuda ? torch::kCUDA : torch::kCPU;

        // Move tensors to the GPU
        states_tensor = states_tensor.to(device);
        new_states_tensor = new_states_tensor.to(device);
        actions_tensor = actions_tensor.to(device);
        rewards_tensor = rewards_tensor.to(device);
        dones_tensor = dones_tensor.to(device);

        torch::Tensor q_values = network.forward(states_tensor);
        torch::Tensor next_target_q_values = target_network.forward(new_states_tensor);
        torch::Tensor next_q_values = network.forward(new_states_tensor);

        actions_tensor = actions_tensor.to(torch::kInt64);

        torch::Tensor q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1);
        torch::Tensor maximum = std::get<1>(next_q_values.max(1));
        torch::Tensor next_q_value = next_target_q_values.gather(1, maximum.unsqueeze(1)).squeeze(1);
        torch::Tensor expected_q_value = rewards_tensor + gamma * next_q_value * (1 - dones_tensor);

        torch::Tensor loss = torch::mse_loss(q_value, expected_q_value);

        dqn_optimizer.zero_grad();
        loss.backward();
        dqn_optimizer.step();

        return loss;
    }


    void Trainer::load_enviroment(int64_t random_seed, std::string rom_path){
        //NOT IMPLEMENTED: to be interfaced with env later

    }

    double Trainer::epsilon_by_frame(int64_t frame_id){
        return epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * frame_id / epsilon_decay);
    }

    // used for convert ale state to tensor state. modify to match the env spec later
    torch::Tensor Trainer::get_tensor_observation(std::vector<unsigned char> state) {
        std::vector<int64_t > state_int;
        state_int.reserve(state.size());

        for (int i=0; i<state.size(); i++){
            state_int.push_back(int64_t(state[i]));
        }

        torch::Tensor state_tensor = torch::from_blob(state_int.data(), {1, 3, 210, 160}); //for cartpole:1,4
        return state_tensor;
    }

    //synchronize target with model
    void Trainer::loadstatedict(torch::nn::Module& model,
                       torch::nn::Module& target_model) {
        torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
        auto new_params = target_model.named_parameters(); // implement this
        auto params = model.named_parameters(true /*recurse*/);
        auto buffers = model.named_buffers(true /*recurse*/);
        for (auto& val : new_params) {
            auto name = val.key();
            auto* t = params.find(name);
            if (t != nullptr) {
                t->copy_(val.value());
            } else {
                t = buffers.find(name);
                if (t != nullptr) {
                    t->copy_(val.value());
                }
            }
        }
    }

    void Trainer::trainloop(int64_t random_seed, std::string rom_path, int64_t num_epochs){
        // load_enviroment(random_seed, rom_path);
        // ActionVect legal_actions = ale.getLegalActionSet();
        // ale.reset_game();
        // std::vector<unsigned char> state;
        // ale.getScreenRGB(state);
        //Randomly generate state for now. interface with env later
        float episode_reward = 0.0;
        std::vector<float> all_rewards;
        std::vector<torch::Tensor> losses;
        auto start = std::chrono::high_resolution_clock::now();
        
        for(int i=1; i<=num_epochs; i++){
            double epsilon = epsilon_by_frame(i);
            auto r = ((double) rand() / (RAND_MAX));
            // torch::Tensor state_tensor = get_tensor_observation(state);
            torch::Tensor state_tensor = torch::rand({1, 4});
            int a;
            if (r <= epsilon){
                // a = legal_actions[rand() % legal_actions.size()];
                a=rand()%2; //randonly generate 0 or 1 for cartpole actions. interface with env according to avail actions later
            }
            else{
                torch::Tensor action_tensor = network.act(state_tensor);
                std::vector<int> legal_actions{ 0,1 };
                int64_t index = action_tensor[0].item<int64_t>(); 
                a = legal_actions[index]; //interface with env according to avail actions later

            }

            // float reward = ale.act(a);
            float reward = (float) rand(); //interface with env according to avail actions later
            episode_reward += reward;
            // std::vector<unsigned char> new_state;
            // ale.getScreenRGB(new_state);
            // torch::Tensor new_state_tensor = get_tensor_observation(new_state);
            torch::Tensor new_state_tensor = torch::rand({1, 4}); //interface with env according to avail actions later
            // bool done = ale.game_over();
            bool done =rand()%2; //interface with env according to avail actions later

            torch::Tensor reward_tensor = torch::tensor(reward); //dim:1
            torch::Tensor done_tensor = torch::tensor(done); //dim:1
            done_tensor = done_tensor.to(torch::kFloat32); //dim:1
            torch::Tensor action_tensor_new = torch::tensor(a); //dim:1

            //interface with replay later
            // buffer.push(state_tensor, new_state_tensor, action_tensor_new, done_tensor, reward_tensor);
            state_tensor = new_state_tensor;
    
            if (done){
                // ale.reset_game();
                // std::vector<unsigned char> state;
                state_tensor = torch::rand({1, 4}); //renew
                // ale.getScreenRGB(state);
                all_rewards.push_back(episode_reward);
                episode_reward = 0.0;
            }

            // if (buffer.size_buffer() >= 10000){
            if (i >= 5000){

                // Suppose sampled random data for now for testbench
                torch::Tensor states_tensor = torch::rand({batch_size, 4},torch::requires_grad());
                torch::Tensor new_states_tensor = torch::rand({batch_size, 4},torch::requires_grad());
                torch::Tensor actions_tensor = torch::rand({batch_size},torch::requires_grad());
                torch::Tensor rewards_tensor = torch::rand({batch_size},torch::requires_grad());
                rewards_tensor = torch::bernoulli(rewards_tensor); //random 0 or 1
                torch::Tensor dones_tensor = torch::rand({batch_size},torch::requires_grad());
                dones_tensor = torch::bernoulli(dones_tensor); //random 0 or 1
                // std::cout<<"states_tensor.sizes:"<<states_tensor.sizes()[0]<<", "<<states_tensor.sizes()[1]<<"\n";
                // std::cout<<"new_states_tensor.sizes:"<<new_states_tensor.sizes()[0]<<", "<<new_states_tensor.sizes()[1]<<"\n";
                // std::cout<<"actions_tensor.sizes:"<<actions_tensor.sizes()[0]<<"\n";
                // std::cout<<"rewards_tensor.sizes:"<<rewards_tensor.sizes()[0]<<"\n";
                // std::cout<<"dones_tensor.sizes:"<<dones_tensor.sizes()[0]<<"\n";
            
                auto start = std::chrono::high_resolution_clock::now();
                torch::Tensor loss = compute_td_loss(batch_size, gamma,
                states_tensor,new_states_tensor,actions_tensor,rewards_tensor,dones_tensor);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << "Time taken by one gradient step of batch size "<<batch_size<<": "
                    << duration.count() << " ms" << std::endl;
            }

            if (i%1000==0){
                std::cout<<"episode_reward: "<<episode_reward<<std::endl;
                loadstatedict(network, target_network);
            }

        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time taken by trainloop function: "
             << duration.count() << " microseconds" << std::endl;


    }


int main() {
    Trainer trainer(4, 2, 100000); 
    trainer.trainloop(123, "/Users/cartpole.bin", 10000);
}