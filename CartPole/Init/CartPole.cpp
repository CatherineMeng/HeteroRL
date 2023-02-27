
#include <torch/types.h>
#include <torch/utils.h>
#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>

#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

// num_repetitions: How many times to repeat the kernel invocation
size_t num_repetitions = 1;
// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<float> FloatVector; 

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
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
 
   torch::Tensor state;
   double reward;
   bool done;
   int step_ = 0;
 
   torch::Tensor getState() {
     return state;
   }
 
   double getReward() {
     return reward;
   }
 
   double isDone() {
     return done;
   }
 
   void reset() {
     state = torch::empty({4}).uniform_(-0.05, 0.05);
     steps_beyond_done = -1;
     step_ = 0;
   }
 
   CartPole() {
     reset();
   }
 
   void step(int action) {
     auto x = state[0].item<float>();
     auto x_dot = state[1].item<float>();
     auto theta = state[2].item<float>();
     auto theta_dot = state[3].item<float>();
 
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
     state = torch::tensor({x, x_dot, theta, theta_dot});
 
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
         AT_ASSERT(false); // Can't do this
       }
     }
     step_++;
   }
 };

//************************************
// Policy for CartPole env on FPGA. Does not depend on state, random
//************************************
void BasicPolicy(queue &q, int in_signal, FloatVector &state_vector,
               int action) {
  // Create the range object for the vectors managed by the buffer.
  // range<1> num_items{a_vector.size()};

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer a_buf(state_vector);

    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    q.submit([&](handler &h) {
      // Create an accessor for each buffer with access permission: read, write or
      // read/write. The accessor is a mean to access the memory in the buffer.
      accessor a(a_buf, h, read_only);
      // accessor b(b_buf, h, read_only);
      h.single_task([=]() { 
      	if (in_signal%2==0)action=1;
      	else action=0;  }
      	);
    });
  // Wait until compute tasks on GPU done
  q.wait();
}

 int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  std::vector<float> rewards;
  FloatVector state_vec;
  state_vec.resize(4, 0);
  auto env = CartPole();
  double running_reward = 10.0;
	for (size_t episode = 0;; episode++) {
		 env.reset();
		 auto state = env.getState();

		 int t = 0;
		 for (; t < 100; t++) {
		   int action = -1;
		   // if (t%2==0) action = 1; //naive policy
		  // generate action given state
		  try {
		    queue q(d_selector, exception_handler);

		    // Print out the device information used for the kernel code.
		    std::cout << "Running on device: "
		              << q.get_device().get_info<info::device::name>() << "\n";

		    // Vector addition in DPC++
		    BasicPolicy(q, t, state_vec, action);
		    std::cout << "action: " << action << "\n";
		  } catch (exception const &e) {
		    std::cout << "An exception is caught for Basic Policy.\n";
		    std::terminate();
		  }
		   env.step(action);
		   state = env.getState();
		   	for (int i=0;i<4;i++){state_vec[i] = state[i].item<float>();}
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
