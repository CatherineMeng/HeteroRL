//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iostream>
#include <vector>

#include <ctime>
#include <cmath>
#include <cstdlib>
#include <random>


// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// Vector size for this example
constexpr size_t StateSpace = 4;

typedef std::vector<float> FloatVector;

float Rand_between(float lo, float hi){
  // printf("hi from Rand_between\n");
  srand (static_cast <unsigned> (time(0)));
  float r3 = lo + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-lo+0.01)));
  return r3;
};

const double kPi = 3.1415926535898;
 
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
    
   FloatVector state;
   // state.resize(4); 
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
    state.resize(4, 0);
     // state = torch::empty({4}).uniform_(-0.05, 0.05);
    for (int i=0;i<4;i++){
      state[i]=Rand_between(-0.05, 0.05);
    }
     steps_beyond_done = -1;
     step_ = 0;
   }
 
   CartPole() {
     reset();
   }
 
   void step(int action) {
     // auto x = state[0].item<float>();
     // auto x_dot = state[1].item<float>();
     // auto theta = state[2].item<float>();
     // auto theta_dot = state[3].item<float>();
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
     // state = torch::tensor({x, x_dot, theta, theta_dot});
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
         // AT_ASSERT(false); // Can't do this
       }
     }
     step_++;
   }
 };

// Forward declare the kernel name in the global scope to reduce name mangling. 
// This is an FPGA best practice that makes it easier to identify the kernel in 
// the optimization reports.
class Policy;


int main() {

  // Set up three vectors and fill two with random values.
  // std::vector<int> vec_a(kSize), vec_b(kSize), vec_r(kSize);
  // for (int i = 0; i < kSize; i++) {
  //   vec_a[i] = rand();
  //   vec_b[i] = rand();
  // }

  std::vector<float> state_vec(StateSpace);
  std::vector<float> action_vec(1);
  std::vector<float> index_vec(1);
  std::vector<float> rewards;
  auto env = CartPole();
  double running_reward = 10.0;

  // Select either:
  //  - the FPGA emulator device (CPU emulation of the FPGA)
  //  - the FPGA device (a real FPGA)
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif

  try {

    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    queue q(device_selector, dpc_common::exception_handler);

    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    {
      
      // Create buffers to share data between host and device.
      // The runtime will copy the necessary data to the FPGA device memory
      // when the kernel is launched.
      buffer buf_a(state_vec);
      buffer buf_b(index_vec);
      buffer buf_r(action_vec);


      // Submit a command group to the device queue.
      q.submit([&](handler& h) {

        // The SYCL runtime uses the accessors to infer data dependencies.
        // A "read" accessor must wait for data to be copied to the device
        // before the kernel can start. A "write no_init" accessor does not.
        accessor a(buf_a, h, read_only);
        accessor b(buf_b, h, read_only);
        accessor r(buf_r, h, write_only, no_init);

        // The kernel uses single_task rather than parallel_for.
        // The task's for loop is executed in pipeline parallel on the FPGA,
        // exploiting the same parallelism as an equivalent parallel_for.
        //
        // The "kernel_args_restrict" tells the compiler that a, b, and r
        // do not alias. For a full explanation, see:
        //    DPC++FPGA/Tutorials/Features/kernel_args_restrict
        h.single_task<Policy>([=]() [[intel::kernel_args_restrict]] {
          r[0] = 1;
          if (b[0]==1) {
            r[0] = 0;
          }
        });
      });

      // The buffer destructor is invoked when the buffers pass out of scope.
      // buf_r's destructor updates the content of vec_r on the host.
      
      for (size_t episode = 0;; episode++) {
         env.reset();
         // printf("hii\n");
         auto state = env.getState();
         int t = 0;
         for (; t < 100; t++) {
           int action = -1;
           if (t%2==0) action = 1; //naive policy
          // generate action given state
                 env.step(action);
           state = env.getState();
           auto reward = env.getReward();
           auto done = env.isDone();

           rewards.push_back(reward);
           if (done)
             break;
         }
         running_reward = running_reward * 0.99 + t * 0.01;
          if (episode % 10 == 0) {
             printf("Episode %i\tLast length: %5d\tAverage length: %.2f\n",
                     episode, t, running_reward);
            }
      }
    }

    // The queue destructor is invoked when q passes out of scope.
    // q's destructor invokes q's exception handler on any device exceptions.
  }
  catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  // Check the results.
  int correct = 0;
  for (int i = 0; i < kSize; i++) {
    if ( vec_r[i] == vec_a[i] + vec_b[i] ) {
      correct++;
    }
  }

  // Summarize and return.
  if (correct == kSize) {
    std::cout << "PASSED: results are correct\n";
  } else {
    std::cout << "FAILED: results are incorrect\n";
  }

  return !(correct == kSize);
}
