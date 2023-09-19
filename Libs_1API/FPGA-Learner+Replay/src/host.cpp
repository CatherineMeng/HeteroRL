//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// On devcloud:
// #include <sycl/sycl.hpp>
// On local install:
#include <CL/sycl.hpp>

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

#include "rmm.hpp"
#include "mlptrain_customizable.hpp"

#include "autorun.hpp"
using namespace sycl;
using namespace std::chrono;

#include "include/exception_handler.hpp"

// === on devcloud ===
// // device selector
// #if FPGA_SIMULATOR
//     auto selector = sycl::ext::intel::fpga_simulator_selector_v;
// #elif FPGA_HARDWARE
//     auto selector = sycl::ext::intel::fpga_selector_v;
// #else  // #if FPGA_EMULATOR
//     auto selector = sycl::ext::intel::fpga_emulator_selector_v;
// #endif

// === on local machine ===
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector selector;
#else
  // The default device selector will select the most performant device.
  default_selector selector;
#endif

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

template <typename Tin, typename Tout>
void DoWorkMultiKernel(sycl::queue& q, std::vector<sibit_io> &in, std::vector<int> &sampled_idx, std::vector<fixed_l3> &out_pr_sampled, std::vector<fixed_l3> &out_pr_insertion,
                        fixed_root &root_pr, 
                        size_t batchsize, size_t iterations);


template<typename T>
void PrintPerformanceInfo(std::string print_prefix, size_t count,
                          std::vector<double>& latency_ms,
                          std::vector<double>& process_time_ms);

void Init_Tree(sycl::queue& q, fixed_root &root_pr);

// the pipes used to produce/consume data
using ProducePipe = ext::intel::pipe<class ProducePipeClass, sibit_io,16>;
using ConsumePipe1 = ext::intel::pipe<class ConsumePipe1Class, int,16>; //sampled ind
using ConsumePipe2 = ext::intel::pipe<class ConsumePipe2Class, fixed_l3,16>; //samapled val
using ConsumePipe3 = ext::intel::pipe<class ConsumePipe3Class, fixed_l3,16>; //get_priority val for insertion


// internal pipes between kernels
using L1_L2_Pipe = ext::intel::pipe<class L1_L2_PipeClass, sibit_io,16>;
using L2_L3_Pipe = ext::intel::pipe<class L2_L3_PipeClass, sibit_io,16>;

// declaring a global instance of this class causes the constructor to be called
// before main() starts, and the constructor launches the kernel.
// fpga_tools::Autorun<Itm1> ar_kernel1{selector, MyAutorun_itm<Itm1, fixed_l1, fixed_root, ProducePipe, L1_L2_Pipe, 1, Lev1_Width>{}};
// fpga_tools::Autorun<Itm2> ar_kernel2{selector, MyAutorun_itm<Itm2, fixed_l2, fixed_l1, L1_L2_Pipe, L2_L3_Pipe, 2, Lev2_Width>{}};
// fpga_tools::Autorun<Itm3> ar_kernel3{selector, MyAutorun_lastlev<Itm3, fixed_l3, fixed_l2, L2_L3_Pipe, ConsumePipe1, ConsumePipe2, ConsumePipe3, Lev3_Width>{}};

int main(int argc, char* argv[]) {

  size_t batchsize=16;
  size_t iterations = 1;
  
  std::cout << "batchsize: " << batchsize << "\n";
  std::cout << "Iterations:           " << iterations-1 << "\n";
  std::cout << "\n";

  bool passed = true;

  try {

    // create the device queue
    sycl::queue q(selector, exception_handler);

    // make sure the device supports USM host allocations
    auto device = q.get_device();

        MyAutorun_itm<Itm1, fixed_l1, fixed_root, ProducePipe, L1_L2_Pipe, 1, Lev1_Width>(q);

         MyAutorun_itm<Itm2, fixed_l2, fixed_l1, L1_L2_Pipe, L2_L3_Pipe, 2, Lev2_Width>(q);

         MyAutorun_lastlev<Itm3,fixed_l3, fixed_l2, L2_L3_Pipe, ConsumePipe1, ConsumePipe2, ConsumePipe3, Lev3_Width>(q);

    // the buffer input and output data
    std::vector<sibit_io> in;
    std::vector<int> out_sampled_idx; //sampling output
    std::vector<fixed_l3> out_sampled_value; //sampling output
    std::vector<fixed_l3> out_insertion_getPr_value; //insertion output

    in.resize(batchsize*4);
    out_sampled_idx.resize(batchsize*4);
    out_sampled_value.resize(batchsize*4);
    out_insertion_getPr_value.resize(batchsize*4);

   
    // ========== TestBench: Init ===========
    fixed_root root_pr=0;
    Init_Tree(q,root_pr);
    
    // ========== TestBench: Insertion - get priority ===========
    // generate the input data for insertion
    // get_pr_value
    for (size_t ii=0; ii<batchsize*4; ii++){
      in[ii].sampling_flag=0;
      in[ii].update_flag=0;
      in[ii].get_priority_flag=1;
      in[ii].init_flag=0;
      in[ii].pr_idx=ii;
    }

    std::cout << "=== Running the get-priority kernel,"<<" batch size= ",batchsize*4<<" ===\n";
    DoWorkMultiKernel<sibit_io,fixed_l3>(q, in, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
    root_pr, batchsize*4, iterations);
    // validate the results 
    printf("out_pr_insertion: ");
    for (size_t ii=0; ii<64; ii++){
      printf("%f ", out_insertion_getPr_value[ii]); //should be all 0 if static init is successful. 
    }    
    std::cout << "\n";

    // ========== TestBench: Insertion - Update ===========
    for (int ii=0; ii<batchsize*4; ii++){
      in[ii].sampling_flag=0;
      in[ii].update_flag=1;
      in[ii].get_priority_flag=0;
      in[ii].init_flag=0;
      in[ii].update_index_array[0]=0;
      in[ii].update_index_array[1]=(ii/K)/K;
      in[ii].update_index_array[2]=ii/K;
      in[ii].update_index_array[3]=ii;
      for (size_t iii=0; iii<D; iii++)in[ii].update_offset_array[iii]=0.1;
    }
    std::cout << "=== Running the update kernel,"<<" batch size= ",batchsize*4<<"\n";
    DoWorkMultiKernel<sibit_io,fixed_l3>(q, in, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
    root_pr, batchsize*4, iterations);
    

    // validate the results 
    std::cout << "Completed the update kernel\n";
    std::cout <<"Root value (updated): "<< root_pr << "\n";//should return 6.4.
    // On the FPGA side: PRINTF in the producer (lev1) should accumulates to 1.6 in the end. 

    // ========== TestBench: Sampling ===========
    // size should be chunks
    fixed_root tb_rand[16]={0.1, 1.0, 1.4, 6.3, 5.0, 3.1, 3.7, 2.0, 6.1, 0.2, 0.9, 1.7, 3.3, 2.8, 4.1, 3.2};
    for (size_t ii=0; ii<batchsize; ii++){
      in[ii].sampling_flag=1;
      in[ii].update_flag=0;
      in[ii].get_priority_flag=0;
      in[ii].init_flag=0;
      in[ii].start=0;
      in[ii].newx=tb_rand[ii];
    }
    std::cout << "Running the Sampling kernel,"<<" batch size= ",batchsize<<" ===\n";
    DoWorkMultiKernel<sibit_io,fixed_l3>(q, in, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
    root_pr, batchsize, iterations);

    // validate the results 
    std::cout << "\nSampled indices results:\n";
    for (size_t ii=0; ii<64; ii++){
      printf("%d ", out_sampled_idx[ii]);
    } 
    // 1st 16 elements should be: {0 9 13 62 49 30 36 19 60 1 8 16 32 27 40 31}.
    std::cout << "\nSampled values results:\n";
    for (size_t ii=0; ii<64; ii++){
      printf("%f ", out_sampled_value[ii]);
    } 
    // 1st 16 elements should all be 0.1. 
    ////////////////////////////////////////////////////////////////////////////

    // free the USM pointers
    // sycl::free(in, q);
    // sycl::free(out_sampled_idx, q);
    // sycl::free(out_sampled_value, q);
    // sycl::free(out_insertion_getPr_value, q);

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

// Tin:sibit_io, Tout: fixed_l3
template <typename Tin, typename Tout>
void DoWorkMultiKernel(sycl::queue& q, std::vector<sibit_io> &in, std::vector<int> &sampled_idx, std::vector<fixed_l3> &out_pr_sampled, std::vector<fixed_l3> &out_pr_insertion,
// Tin* in, int* sampled_idx, Tout* out_pr_sampled, Tout* out_pr_insertion,
                        fixed_root &root_pr, 
                        size_t batchsize, size_t iterations) {
  // timing data
  // std::vector<double> latency_ms(iterations);
  double process_time_ms(iterations);

  for (size_t i = 0; i < iterations; i++) {
 
    for (size_t i = 0; i < batchsize; i++){
         if (in[i].update_flag==1){
          // std::cout<<"root_ptr = "<<root_pr<<" ";
          root_pr+=in[i].update_offset_array[0];
        }
    }

    auto start = high_resolution_clock::now();


    event p_e = Submit_Producer<ProducePipe>(q, in, batchsize);   
    event c_e = Submit_Consumer<ConsumePipe1, ConsumePipe2, ConsumePipe3, fixed_l3>(q, 
                                                sampled_idx,
                                                out_pr_sampled,
                                                out_pr_insertion,
                                                batchsize);

    p_e.wait();    // producer
    c_e.wait();   // consumer

    auto end = high_resolution_clock::now();

    // compute latency and processing time
    duration<double, std::milli> process_time = end - start;
    process_time_ms = process_time.count();
    double avg_tp_samples_s = batchsize / (process_time_ms * 1e-3);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " Total latency:  " << process_time_ms << " ms\n";
    std::cout << " Troughput:  " << avg_tp_samples_s << " samples/s\n";
  }

  // compute and print timing information
  // PrintPerformanceInfo<T>("Multi-kernel",total_count, latency_ms, process_time_ms);
}

void Init_Tree(sycl::queue& q, fixed_root &root_pr){
 
    size_t batchsize = 2;       

    std::cout << "=== Running the kernel for initializing tree ===\n";
    // sibit_io *in;
    // int *sampled_idx;
    // fixed_l3 *out_pr_sampled;
    // fixed_l3 *out_pr_insertion;
    std::vector<sibit_io> in;
    std::vector<int> sampled_idx; //sampling output
    std::vector<fixed_l3> out_pr_sampled; //sampling output
    std::vector<fixed_l3> out_pr_insertion; //insertion output
    // if ((sampled_idx = malloc_host<int>(2, q)) == nullptr ||
    //     (out_pr_sampled = malloc_host<fixed_l3>(2, q)) == nullptr ||
    //     (out_pr_insertion = malloc_host<fixed_l3>(2, q)) == nullptr) {
    //   std::cerr << "ERROR: could not allocate space for 'out'\n";
    //   std::terminate();
    // }
    // if ((in = malloc_host<sibit_io>(2, q)) == nullptr) {
    //   std::cerr << "ERROR: could not allocate space for 'in'\n";
    //   std::terminate();
    // }
    in.resize(2);
    sampled_idx.resize(2);
    out_pr_sampled.resize(2);
    out_pr_insertion.resize(2);
    // ======== Test ========
    in[0].sampling_flag=0;
    in[0].update_flag=0;
    in[0].get_priority_flag=1;
    in[0].pr_idx=0;
    in[0].init_flag=1;
    in[1].sampling_flag=0;
    in[1].update_flag=0;
    in[1].get_priority_flag=1;
    in[1].pr_idx=1; //both get_priority should return 0
    in[1].init_flag=1; 

    std::queue<std::pair<event,event>> event_q;

    root_pr=0;


    // auto events = Submit_Intermediate_SiblingItr<Itm1,fixed_l2,fixed_l3,L1_L2_Pipe,L2_L3_Pipe,2,16>(q, chunk_size*chunks);

      
        event p_e = Submit_Producer<ProducePipe>(q, in, batchsize);   
        event c_e = Submit_Consumer<ConsumePipe1, ConsumePipe2, ConsumePipe3, fixed_l3>(q, 
                                                   sampled_idx,
                                                   out_pr_sampled,
                                                   out_pr_insertion,
                                                   batchsize);        


        p_e.wait();    // producer
        c_e.wait();   // consumer

    // events.wait();
    std::cout<<"out_pr_insertion for get priority:"<<out_pr_insertion[0]<<" "<<out_pr_insertion[1]<<"\n";
    // should print 0 0

    printf("Init tree done. \n\n");
}

// a helper function to compute and print the performance info
template<typename T>
void PrintPerformanceInfo(std::string print_prefix, size_t count,
                          std::vector<double>& latency_ms,
                          std::vector<double>& process_time_ms) {
  // compute the input size in MB
  double input_size_megabytes = (sizeof(T) * count) * 1e-6;

  // compute the average latency and processing time
  double iterations = latency_ms.size() - 1;
  double avg_latency_ms = std::accumulate(latency_ms.begin() + 1,
                                          latency_ms.end(),
                                          0.0) / iterations;
  double avg_processing_time_ms = std::accumulate(process_time_ms.begin() + 1,
                                                  process_time_ms.end(),
                                                  0.0) / iterations;

  // compute the throughput
  double avg_tp_mb_s = input_size_megabytes / (avg_processing_time_ms * 1e-3);

  // print info
  std::cout << std::fixed << std::setprecision(4);
  std::cout << print_prefix
            << " average latency:           " << avg_latency_ms << " ms\n";
  std::cout << print_prefix
            << " average throughput:        " << avg_tp_mb_s  << " MB/s\n";
}

