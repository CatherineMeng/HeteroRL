//==============================================================
// Author: Yuan Meng
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

// #include "matmul.hpp"
#include "mlptrain.hpp"

#include "autorun.hpp"

using namespace sycl;
using namespace std::chrono;

#include "include/exception_handler.hpp"

// device selector
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
// queue properties to enable profiling
property_list prop_list { property::queue::enable_profiling() };

void DoWorkMultiKernel(sycl::queue& q, StateConcate *state_in_buf, W1Fmt *w1_buf, W2Fmt *w2_buf, act_fmt *bias1_buf, act_fmt *bias2_buf, RDone *rdone_buf,
                        W2TranspFmt *w2t_buf, L2AG *wg1_buf, L3AG *wg2_buf, act_fmt *biasg1_buf, act_fmt *biasg2_buf,
                        size_t chunks, size_t chunk_size, size_t total_count,
                        size_t inflight_kernels, size_t iterations);
// The following functions are for validating results on the host
void forwardPass(const float* input, const float* input_nt, const float* hiddenBiases,
                 const std::vector<std::vector<float>>& hiddenWeights, const float* outputBiases,
                 const std::vector<std::vector<float>>& outputWeights,
                 float* hiddenOutput, float* hiddenOutput_nt, float* output, float* output_nt);
void backwardPass(const float* input, const float* hiddenOutput,
                  const float* output, const float* targetOutput, const RDone rd,
                  const std::vector<std::vector<float>>& hiddenWeights,
                  const std::vector<std::vector<float>>& outputWeights,
                  std::vector<std::vector<float>>& hiddenWeightGradients,
                  std::vector<std::vector<float>>& outputWeightGradients,
                  float* hiddenBiasGradients, float* outputBiasGradients);
void updateWeightsAndBiases(float learningRate, 
                            std::vector<std::vector<float>>& hiddenWeights,
                            std::vector<std::vector<float>>& outputWeights,
                            float* hiddenBiases, float* outputBiases,
                            const std::vector<std::vector<float>>& hiddenWeightGradients,
                            const std::vector<std::vector<float>>& outputWeightGradients,
                            const float* hiddenBiasGradients, const float* outputBiasGradients);

// the pipes used to produce/consume data
using ProducePipe = ext::intel::pipe<class ProducePipeClass, act_fmt,16>;
using ConsumePipe = ext::intel::pipe<class ConsumePipe1Class, act_fmt,16>; //sampled ind


// declaring a global instance of this class causes the constructor to be called
// before main() starts, and the constructor launches the kernel.
fpga_tools::Autorun<MM_FW1> ar_kernel1{selector, MyAutorun_MMFW<MM_FW1, W1Fmt, StateConcate,L2ItmConcate, L2AG,
         SinPipe, L1FWSigPipe, ReadW1Pipe, ReadB1Pipe, L12Pipe, A1Pipe, L1, L2>{}};

fpga_tools::Autorun<MM_FW2> ar_kernel2{selector, MyAutorun_MMFW_OL<MM_FW2, W2Fmt, L2ItmConcate, L3ItmConcate,
         L12Pipe, L2FWSigPipe, ReadW2Pipe, ReadB2Pipe, L23Pipe, L2, L3>{}};

fpga_tools::Autorun<OBJ> ar_kernel3{selector, MyAutorun_OBJ<OBJ, L3ItmConcate, L3AG, 
         L23Pipe, RDonePipe, L32Pipe, D2Pipe, L3>{}};

fpga_tools::Autorun<MM_BW> ar_kernel4{selector, MyAutorun_MMBW_OL <MM_BW, W2TranspFmt, L3AG, L2AG,
         L32Pipe, L2BWSigPipe, ReadW2bwPipe, D1Pipe, L3, L2>{}};

fpga_tools::Autorun<WA1> ar_kernel5{selector, MyAutorun_MMWA <WA1, L1AG, L2AG, L2AG,
         A0Pipe, D1Pipe, writeW1Pipe, writeB1Pipe, L1, L2>{}};
fpga_tools::Autorun<WA2> ar_kernel6{selector, MyAutorun_MMWA <WA2, L2AG, L3AG, L3AG,
         A1Pipe, D2Pipe, writeW2Pipe, writeB2Pipe, L2, L3>{}};



// '''
// '''

int main(int argc, char* argv[]) {
  // default values
  #if defined(FPGA_EMULATOR)
    size_t chunks = 1;        
    size_t chunk_size = 1;    // 1
    size_t iterations = 1; //ToDo: This could be the batch size?
  #elif defined(FPGA_SIMULATOR)
    size_t chunks = 1;         
    size_t chunk_size = 1;    // 1
    size_t iterations = 1; //ToDo: This could be the batch size?
  #else
    size_t chunks = 8;       
    size_t chunk_size = 1;   // 1
    size_t iterations = 1; //ToDo: This could be the batch size?
  #endif

  // This is the number of kernels we will have in the queue at a single time.
  // If this number is set too low (e.g. 1) then we don't take advantage of
  // fast kernel relaunch (see the README). If this number is set to high,
  // then the first kernel launched finishes before we are done launching all
  // the kernels and therefore throughput is decreased.
  size_t inflight_kernels = 1;
  // compute the total number of elements
  size_t total_count = chunks * chunk_size;

  std::cout << "# Chunks:             " << chunks << "\n";
  std::cout << "Chunk size:          " << chunk_size << "\n";
  std::cout << "Total count:          " << total_count << "\n";
  std::cout << "Iterations:           " << iterations-1 << "\n";
  std::cout << "\n";

  bool passed = true;

  try {

    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler, prop_list);


    // make sure the device supports USM host allocations
    auto device = q.get_device();
    if (!device.get_info<info::device::usm_host_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
      std::terminate();
    }
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // the USM input and output data
    // Type *in, *out;
    StateConcate *state_in_buf;
    W1Fmt *w1_buf;
    W2Fmt *w2_buf;
    act_fmt *bias1_buf;
    act_fmt *bias2_buf;
    RDone *rdone_buf;

    W2TranspFmt *w2t_buf;

    L2AG *wg1_buf;
    L3AG *wg2_buf;
    act_fmt *biasg1_buf;
    act_fmt *biasg2_buf;
    // assume batch size 8
    if ((state_in_buf = malloc_host<StateConcate>(chunks*chunk_size, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'state_in_buf'\n";
      std::terminate();
    }
    if ((w1_buf = malloc_host<W1Fmt>(L2, q)) == nullptr || (w2_buf = malloc_host<W2Fmt>(L3, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for w1/w2_buf\n";
      std::terminate();
    }
    if ((bias1_buf = malloc_host<act_fmt>(L2, q)) == nullptr || (bias2_buf= malloc_host<act_fmt>(L3, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for b1/b2_buf\n";
      std::terminate();
    }
    if ((rdone_buf= malloc_host<RDone>(chunks*chunk_size, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'RDone buf'\n";
      std::terminate();
    }
    if ((w2t_buf= malloc_host<W2TranspFmt>(L2, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'w2t_buf'\n";
      std::terminate();
    }
    if ((wg1_buf = malloc_host<L2AG>(L1, q)) == nullptr || (wg2_buf = malloc_host<L3AG>(L2, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for wg1/wg2_buf\n";
      std::terminate();
    }
    if ((biasg1_buf = malloc_host<act_fmt>(L2, q)) == nullptr || (biasg2_buf= malloc_host<act_fmt>(L3, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for bg1/bg2_buf\n";
      std::terminate();
    }


    // ========== TestBench ===========

    std::cout << "Running the MLP train Host code\n";

    //Initialize weight & out for software emulation comparison
    // Define the learning rate
    float learningRate = 0.1f;

    // Define the input array, the hidden and output arrays
    float input[L1] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float input_nt[L1] = {1.2f, 2.0f, 1.0f, 3.2f, 5.0f, 6.3f, 7.0f, 5.0f,
                       9.0f, 7.0f, 8.7f, 12.0f, 1.9f, 4.0f, 5.7f, 12.0f};
    float hiddenOutput[L2] = {0.0f};
    float hiddenOutput_nt[L2] = {0.0f};
    float output[L3] = {0.0f};
    float output_nt[L3] = {0.0f};

    // Define the biases
    float hiddenBiases[L2] = {0.1f};
    float outputBiases[L3] = {0.2f};

    // Initialize the weights (as vectors of vectors)
    std::vector<std::vector<float>> hiddenWeights(L1, std::vector<float>(L2, 1.0f));
    std::vector<std::vector<float>> outputWeights(L2, std::vector<float>(L3, 1.0f));

    // Define the target output (update with RL obj: this is generated by Q_next(s_next))
    // float targetOutput[L3] = {0.9f, 0.1f, 0.8f, 0.2f, 0.7f, 0.3f, 0.6f, 0.4f};
    RDone rd;
    rd.done=0;
    rd.r=2;

    // Define the weight gradients (as vectors of vectors)
    std::vector<std::vector<float>> hiddenWeightGradients(L1, std::vector<float>(L2, 0.0f));
    std::vector<std::vector<float>> outputWeightGradients(L2, std::vector<float>(L3, 0.0f));

    // Define the bias gradients
    float hiddenBiasGradients[L2] = {0.0f};
    float outputBiasGradients[L3] = {0.0f};


    // Perform the forward pass
    forwardPass(input, input_nt, hiddenBiases, hiddenWeights, outputBiases, outputWeights, hiddenOutput,hiddenOutput_nt, output, output_nt);

    // Perform the backward pass
    backwardPass(input, hiddenOutput, output, output_nt, rd, hiddenWeights, outputWeights,
                 hiddenWeightGradients, outputWeightGradients, hiddenBiasGradients, outputBiasGradients);

    // Update the weights and biases
    updateWeightsAndBiases(learningRate, hiddenWeights, outputWeights, hiddenBiases, outputBiases,
                           hiddenWeightGradients, outputWeightGradients, hiddenBiasGradients, outputBiasGradients);

    // ========== validate the results on FPGA ========== 
    //Initialize inputs for FPGA kernel

    rdone_buf[0].r=2;
    rdone_buf[0].done=0;
    for (size_t i=0; i<L1; i++) {
      state_in_buf[0].s[i]=input[i]; //replace 0 with sub-batch index later on
      state_in_buf[0].snt[i]=input_nt[i];
    }
    // Define the biases
    for (size_t i=0; i<L2; i++) {
      bias1_buf[i]=hiddenBiases[i];
      biasg1_buf[i]=0;
    }
    for (size_t i=0; i<L3; i++) {
      bias2_buf[i]=outputBiases[i];
      biasg2_buf[i]=0;
    }

    // Initialize the weights
    for (size_t i=0; i<L1; i++) {
      for (size_t j=0; j<L2; j++){
        w1_buf[j].w[i]=hiddenWeights[i][j];
        wg1_buf[i].s[j]=0;
      }
    }
    for (size_t i=0; i<L2; i++) {
      for (size_t j=0; j<L3; j++){
        w2_buf[j].w[i]=outputWeights[i][j];
        w2t_buf[i].w[j]=outputWeights[i][j];
        wg2_buf[i].s[j]=0;
      }
    }

    std::cout << "Running the MLP train FPGA kernel\n";
    
    // std::cout << "hi1\n";std::cout << "hi2\n";std::cout << "hi3\n";
    DoWorkMultiKernel(q, state_in_buf, w1_buf, w2_buf, bias1_buf, bias2_buf, rdone_buf, w2t_buf, 
    wg1_buf, wg2_buf, biasg1_buf, biasg2_buf, 
    chunks, chunk_size, total_count, inflight_kernels, iterations);


    // for (size_t ii=0; ii<8; ii++){
    //     if (out[ii]!=out_host[ii]){passed=false;}
    // }
    std::cout << "Completed the MLP gradient update kernel\n";

    // free the USM pointers
    sycl::free(state_in_buf, q);
    sycl::free(w1_buf, q);
    sycl::free(w2_buf, q);
    sycl::free(bias1_buf, q);
    sycl::free(bias2_buf, q);
    sycl::free(rdone_buf, q);
    sycl::free(w2t_buf, q);
    sycl::free(wg1_buf, q);
    sycl::free(wg2_buf, q);
    sycl::free(biasg1_buf, q);
    sycl::free(biasg2_buf, q);
  } catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
      std::cerr << "If you are targeting the FPGA simulator, compile with "
                   "-DFPGA_SIMULATOR.\n";
    }
    std::terminate();
  }

  if(passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}

// Tin:act_fmt, Tout: act_fmt
// template <typename Tin, typename Tout>
// template <typename T>
void DoWorkMultiKernel(sycl::queue& q, StateConcate *state_in_buf, W1Fmt *w1_buf, W2Fmt *w2_buf, act_fmt *bias1_buf, act_fmt *bias2_buf, RDone *rdone_buf,
                        W2TranspFmt *w2t_buf, L2AG *wg1_buf, L3AG *wg2_buf, act_fmt *biasg1_buf, act_fmt *biasg2_buf,
                        size_t chunks, size_t chunk_size, size_t total_count,
                        size_t inflight_kernels, size_t iterations) {
  // timing data
//   std::vector<double> latency_ms(iterations);
//   std::vector<double> process_time_ms(iterations);

  size_t in_chunk = 0; // count the number of chunks for which kernels have been started
  size_t out_chunk = 0; // count the number of chunks for which kernels have finished 
  // use a queue to track the kernels in flight
  std::queue<std::pair<event,event>> event_q;
  for (size_t i = 0; i < iterations; i++) {
    // std::cout <<"in interation i = "<<i<<"\n";
    // reset the output data to catch any untouched data
    // std::fill_n(out, NL, -1);
    // reset counters
    in_chunk = 0;
    out_chunk = 0;
    // clear the queue
    std::queue<std::pair<event,event>> clear_q;

    std::swap(event_q, clear_q);

    do {
      // if we still have kernels to launch, launch them in here
      if (in_chunk < chunks) {
        // std::cout <<"in_chunk: "<<in_chunk<<"\n";
        // launch the producer/consumer pair for the next chunk of data
        size_t chunk_offset = in_chunk*chunk_size;

        // event p_e = Submit_Producer<ProducePipe>(q, in + chunk_offset, chunk_size);   
        // event c_e = Submit_Consumer<ConsumePipe>(q, out + chunk_offset, chunk_size);
        event p_e1 = Submit_Producer<SinPipe,ReadW1Pipe,ReadW2Pipe,
                                    ReadB1Pipe,ReadB2Pipe,L1FWSigPipe,L2FWSigPipe,A0Pipe,RDonePipe,ReadW2bwPipe,L2BWSigPipe>
                                    (q, state_in_buf + chunk_offset, w1_buf + chunk_offset, w2_buf + chunk_offset, bias1_buf + chunk_offset, bias2_buf + chunk_offset, rdone_buf + chunk_offset, w2t_buf + chunk_offset, chunk_size, 1); //rest of chunks is 0   
        // event p_e2 = Submit_Producer_BW<ReadW2bwPipe,L2BWSigPipe>(q, w2t_buf, 1); //rest of chunks is 0 
        event p_e3 = Submit_Consumer<writeW1Pipe,writeW2Pipe,writeB1Pipe,writeB2Pipe>(q, wg1_buf, wg2_buf, biasg1_buf, biasg2_buf, chunk_size);
        // push the kernel event into the queue
        event_q.push(std::make_pair(p_e1, p_e3));
        // event_q.push(std::make_tuple(p_e1,p_e2,p_e3));

        // if this is the first chunk, track the time
        // if (in_chunk == 0) first_data_in = high_resolution_clock::now();
        in_chunk++;
      }

      // wait on the oldest kernel to finish if any of these conditions are met:
      //    1) there are a certain number kernels in flight
      //    2) all of the kernels have been launched
      //
      // NOTE: 'inflight_kernels' is now the number of inflight
      // producer/consumer kernel pairs
      std::cout <<"here "<<"\n";
      if ((event_q.size() >= inflight_kernels) || (in_chunk >= chunks)) {
        // grab the oldest kernel event we are waiting on
        auto event_pair = event_q.front();
        event_q.pop();
        std::cout <<"event_q.pop(); "<<"\n";

        // wait on the producer/consumer kernel pair to finish
        event_pair.first.wait();    // producer
        std::cout <<"producer completed "<<"\n";
        event_pair.second.wait();   // consumer
        std::cout <<"consumer completed "<<"\n";
        // std::get<0>(event_pair).wait();
        // std::get<1>(event_pair).wait();
        // std::get<2>(event_pair).wait();

        // track the time if this is the first producer/consumer pair
        // if (out_chunk == 0) first_data_out = high_resolution_clock::now();

        // at this point the first 'out_chunk' chunks are ready to be
        // processed on the host
        out_chunk++;
      }
    } while(out_chunk < chunks);

    auto end = high_resolution_clock::now();

  }

  // compute and print timing information
  // PrintPerformanceInfo<T>("Multi-kernel",total_count, latency_ms, process_time_ms);
}

#include <iostream>
#include <vector>
#include <cmath>

// Activation function (ReLU)
float relu(float x) {
    return std::max(0.0f, x);
}

// Forward pass. nt means next state
void forwardPass(const float* input, const float* input_nt, const float* hiddenBiases,
                 const std::vector<std::vector<float>>& hiddenWeights, const float* outputBiases,
                 const std::vector<std::vector<float>>& outputWeights,
                 float* hiddenOutput, float* hiddenOutput_nt, float* output, float* output_nt) {
    // Calculate hidden layer output
    for (int i = 0; i < L2; i++) {
        float sum = hiddenBiases[i];
        float sum_nt = hiddenBiases[i];
        for (int j = 0; j < L1; j++) {
            sum += input[j] * hiddenWeights[j][i];
            sum_nt += input_nt[j] * hiddenWeights[j][i];
        }
        hiddenOutput[i] = relu(sum);
        hiddenOutput_nt[i] = relu(sum_nt); 
    }

    // Calculate output layer output
    for (int i = 0; i < L3; i++) {
        float sum = outputBiases[i];
        float sum_nt = hiddenBiases[i];
        for (int j = 0; j < L2; j++) {
            sum += hiddenOutput[j] * outputWeights[j][i];
            sum_nt += hiddenOutput_nt[j] * outputWeights[j][i];
        }
        output[i] = relu(sum);
        output_nt[i] = relu(sum);
    }
}

// Backward pass
// output: Q values from FW, targetOutput: Q_nt values from FW.
void backwardPass( const float* input, const float* hiddenOutput,
                  const float* output, const float* targetOutput, const RDone rd,
                  const std::vector<std::vector<float>>& hiddenWeights,
                  const std::vector<std::vector<float>>& outputWeights,
                  std::vector<std::vector<float>>& hiddenWeightGradients,
                  std::vector<std::vector<float>>& outputWeightGradients,
                  float* hiddenBiasGradients, float* outputBiasGradients) {
    
    float gamma = 0.3;
    float maxQnt = -9999;
    for (int i = 0; i < L3; i++) {
      if (targetOutput[i]>maxQnt){
        maxQnt=targetOutput[i];
      }
    }
    // Calculate output layer gradients (OBJ)
    for (int i = 0; i < L3; i++) {
      // rdone.r + (1-rdone.done) * gamma * maxQsnt - Qs.s[i];
        // float errorSignal = (targetOutput[i] - output[i]) * (output[i] > 0 ? 1 : 0); (update with RL obj)
        float errorSignal = (rd.r+(1-rd.done)*gamma*maxQnt - output[i]) * (output[i] > 0 ? 1 : 0);
        outputBiasGradients[i] = errorSignal;
        for (int j = 0; j < L2; j++) { //(WA2)
            outputWeightGradients[j][i] = errorSignal * hiddenOutput[j];
        }
    }

    // Calculate hidden layer gradients 
    for (int i = 0; i < L2; i++) {
        float errorSignal = 0.0f;
        for (int j = 0; j < L3; j++) { // (BW)
            errorSignal += (rd.r+(1-rd.done)*gamma*maxQnt - output[i]) * (output[i] > 0 ? 1 : 0) * outputWeights[i][j];
        }
        errorSignal *= (hiddenOutput[i] > 0 ? 1 : 0);
        hiddenBiasGradients[i] = errorSignal;
        for (int j = 0; j < L1; j++) { //(WA1)
            hiddenWeightGradients[j][i] = errorSignal * input[j];
        }
    }
}

// Update weights and biases
void updateWeightsAndBiases(float learningRate, 
                            std::vector<std::vector<float>>& hiddenWeights,
                            std::vector<std::vector<float>>& outputWeights,
                            float* hiddenBiases, float* outputBiases,
                            const std::vector<std::vector<float>>& hiddenWeightGradients,
                            const std::vector<std::vector<float>>& outputWeightGradients,
                            const float* hiddenBiasGradients, const float* outputBiasGradients) {
    // Update output layer weights and biases
    for (int i = 0; i < L3; i++) {
        outputBiases[i] += learningRate * outputBiasGradients[i];
        for (int j = 0; j < L2; j++) {
            outputWeights[j][i] += learningRate * outputWeightGradients[j][i];
        }
    }

    // Update hidden layer weights and biases
    for (int i = 0; i < L2; i++) {
        hiddenBiases[i] += learningRate * hiddenBiasGradients[i];
        for (int j = 0; j < L1; j++) {
            hiddenWeights[j][i] += learningRate * hiddenWeightGradients[j][i];
        }
    }
}

