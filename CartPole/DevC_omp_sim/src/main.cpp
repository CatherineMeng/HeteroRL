//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iomanip>  // setprecision library
#include <iostream>
#include <chrono>

#include <cmath>
#include <cstdlib>
#include <random>

#include <vector>
#include <string>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
// #include "dpc_common.hpp"
typedef std::vector<float> FloatVector; 
typedef std::vector<std::vector<float>> StateVector; 
typedef std::vector<int> IntVector; 
typedef std::vector<bool> BoolVector; 
const double kPi = 3.1415926535898;
int num_rollout=4;
int traj_limit=200;
int state_space=4;
 
 class CartPole {
   // Translated from openai/gym's cartpole.py
  public:
   double gravity = 9.8;
   double masscart = 1.0;
   double masspole = 0.1;
   double total_mass = (masspole + masscart);
   double length = 0.5; // actually half the pole's length;
   double polemass_length = (masspole * length);
   double force_mag = 10.0;
   double tau = 0.02; // seconds between state updates;
 
   // Angle at which to fail the episode
   double theta_threshold_radians = 12 * 2 * kPi / 360;
   double x_threshold = 2.4;
   int steps_beyond_done = -1;
 
  // torch::Tensor state;
   FloatVector state;
   double reward;
   bool done;
   int step_ = 0;
 
   FloatVector getState() {
     return state;
   }
 
   double getReward() {
     return reward;
   }
 
   double isDone() {
     return done;
   }
 
   void reset() {
    //  state = torch::empty({4}).uniform_(-0.05, 0.05);
    float LO=-0.05;
    float HI=0.05;
    for (int i=0;i<4;i++){
        state[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    }
     steps_beyond_done = -1;
     step_ = 0;
   }
 
   CartPole() {
    state.resize(4);
     reset();
   }
 
   void step(int action) {
    //  auto x = state[0].item<float>();
    //  auto x_dot = state[1].item<float>();
    //  auto theta = state[2].item<float>();
    //  auto theta_dot = state[3].item<float>();
     auto x = state[0];
     auto x_dot = state[1];
     auto theta = state[2];
     auto theta_dot = state[3];
 
     auto force = (action == 1) ? force_mag : -force_mag;
     auto costheta = std::cos(theta);
     auto sintheta = std::sin(theta);
     auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta) /
         total_mass;
     auto thetaacc = (gravity * sintheta - costheta * temp) /
         (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
     auto xacc = temp - polemass_length * thetaacc * costheta / total_mass;
 
     x = x + tau * x_dot;
     x_dot = x_dot + tau * xacc;
     theta = theta + tau * theta_dot;
     theta_dot = theta_dot + tau * thetaacc;
    //  state = torch::tensor({x, x_dot, theta, theta_dot});
    state[0]=x;
    state[1]=x_dot;
    state[2]=theta;
    state[3]=theta_dot;
 
     done = x < -x_threshold || x > x_threshold ||
         theta < -theta_threshold_radians || theta > theta_threshold_radians ||
         step_ > 200;
 
     if (!done) {
       reward = 1.0;
     } else if (steps_beyond_done == -1) {
       // Pole just fell!
       steps_beyond_done = 0;
       reward = 0;
     } else {
       if (steps_beyond_done == 0) {
        //  AT_ASSERT(false); // Can't do this
        reward = 0;
        return;
       }
     }
     step_++;
   }
 };

// class Replay{
//   public:
//     FloatVector replay_state;
//     FloatVector replay_action;
//     FloatVector replay_reward;
//     IntVector replay_done;
//    Replay() {
//     replay_state.resize(traj_limit*num_rollout);
//     replay_action.resize(traj_limit*num_rollout);
//     replay_reward.resize(traj_limit*num_rollout);
//     replay_done.resize(traj_limit*num_rollout);
//     //  reset();
//    }
// };

int policy(const FloatVector &state_vector, const FloatVector &param_vector) {
  int action=0; 
  float wsum=0.0;
  for (int i=0;i<4;i++){
    wsum+=state_vector[i]*param_vector[i];
  }
  if (wsum<0)action=0;
  else action=1;
  return action;
}

// cpu_seq_calc_pi is a simple sequential CPU routine
// that calculates all the slices and then
// does a reduction.
/* float cpu_seq_calc_pi(int num_steps) {
  float step = 1.0 / (float)num_steps;
  float x;
  float pi;
  float sum = 0.0;
  for (int i = 1; i < num_steps; i++) {
    x = ((float)i - 0.5f) * step;
    sum = sum + 4.0f / (1.0f + x * x);
  }
  pi = sum * step;
  return pi;
} */
void seq_sim(StateVector &replay_state,FloatVector &replay_action,
            FloatVector &replay_reward,BoolVector &replay_done){
  for (size_t episode = 0; episode<num_rollout; episode++) {
    auto env = CartPole();
    env.reset();
		auto state = env.getState();
    FloatVector param_vec;
    param_vec.resize(state_space);
    std::cout<<"Episode "<<episode<<" params:";
    for (int i=0; i<state_space; i++){
      param_vec[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      std::cout<<" "<<param_vec[i];
    }
    std::cout<<" \n";
    int totalreward=0;
    for (size_t step=0; step<traj_limit; step++){
      replay_state[episode*traj_limit+step]=state;
      int action=policy(state,param_vec);
      env.step(action);
      state = env.getState();
      auto reward = env.getReward();
      auto done = env.isDone();
      replay_action[episode*traj_limit+step]=action;
      replay_reward[episode*traj_limit+step]=reward;
      replay_done[episode*traj_limit+step]=done;
      totalreward+=reward;
      if (done){break;}
    }
    std::cout<<"Total rewards: "<<totalreward<<"\n\n";
  }
}

void openmp_host_sim(StateVector &replay_state,FloatVector &replay_action,
            FloatVector &replay_reward,BoolVector &replay_done) {
  #pragma omp parallel for shared(replay_state,replay_action,replay_reward,replay_done)
  for (size_t tid = 0; tid<num_rollout; tid++) {
    auto env = CartPole();
    env.reset();
		FloatVector state = env.getState();
    FloatVector param_vec;
    param_vec.resize(state_space);
    std::cout<<"Thread "<<tid<<" params:";
    for (int i=0; i<state_space; i++){
      param_vec[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      std::cout<<" "<<param_vec[i];
    }
    std::cout<<" \n";
    int totalreward=0;
    for (size_t step=0; step<traj_limit; step++){
      replay_state[tid*traj_limit+step]=state;
      int action=policy(state,param_vec);
      env.step(action);
      state = env.getState();
      auto reward = env.getReward();
      auto done = env.isDone();
      replay_action[tid*traj_limit+step]=action;
      replay_reward[tid*traj_limit+step]=reward;
      replay_done[tid*traj_limit+step]=done;
      totalreward+=reward;
      if (done){break;}
    }
    std::cout<<"Total rewards: "<<totalreward<<"\n\n";
  }
}

int main(int argc, char** argv) {
    StateVector replay_state;
    FloatVector replay_action;
    FloatVector replay_reward;
    BoolVector replay_done;
    replay_state.resize(traj_limit*num_rollout);
    for (size_t i=0;i<traj_limit*num_rollout;i++){
      replay_state[i].resize(state_space);
    }
    replay_action.resize(traj_limit*num_rollout);
    replay_reward.resize(traj_limit*num_rollout);
    replay_done.resize(traj_limit*num_rollout);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  seq_sim(replay_state,replay_action,replay_reward,replay_done);
  // auto stop = T.Elapsed();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  
  std::cout << "Cpu Seq calc: \t\t";
  std::cout << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms"
            << "\n";

  begin = std::chrono::steady_clock::now();
  // dpc_common::TimeInterval T2;
  openmp_host_sim(replay_state,replay_action,replay_reward,replay_done);
  end = std::chrono::steady_clock::now();
  // auto stop2 = T2.Elapsed();
  std::cout << "Host OpenMP:\t\t";
  std::cout << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms"
            << "\n";

  // // dpc_common::TimeInterval T3;
  // begin = std::chrono::steady_clock::now();
  // pi = openmp_device_calc_pi(num_steps);
  // // auto stop3 = T3.Elapsed();
  // end = std::chrono::steady_clock::now();
  // std::cout << "Offload OpenMP:\t\t";
  // std::cout << std::setprecision(3) << "PI =" << pi;
  // std::cout << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms"
  //           << "\n";

  std::cout << "success\n";
  return 0;
}
