#include <torch/torch.h>

struct DQN : torch::nn::Module{
    DQN(int64_t state_dim, int64_t num_actions) :
            linear1(torch::nn::Linear(state_dim, 32)),
            output(torch::nn::Linear(32, num_actions)){}
            //each action is associated with a unique q value
            // for cartpole, state_dim=4, num_actions=2

    torch::Tensor forward(torch::Tensor input) {
        input = torch::relu(linear1(input));
        input = output(input);
        return input;
    }

    torch::Tensor act(torch::Tensor state){
        torch::Tensor q_value = forward(state);
        torch::Tensor action = std::get<1>(q_value.max(1));
        return action;
    }
    torch::nn::Linear linear1, output;
};

void xavier_init(torch::nn::Module& module) {
	torch::NoGradGuard noGrad;
	if (auto* linear = module.as<torch::nn::Linear>()) {
		torch::nn::init::xavier_normal_(linear->weight);
		torch::nn::init::constant_(linear->bias, 0.01);
	}
}

int main() {
    DQN q_network(4,2);
    xavier_init(q_network);
    torch::Tensor rand_state = torch::rand({16, 4}); //1,4?
    torch::Tensor action_tensor = q_network.act(rand_state);
    std::cout <<"action_tensor with shape "<< action_tensor.sizes()[0]
    <<","<<action_tensor.sizes()[1]<<": "<< action_tensor << std::endl;
}