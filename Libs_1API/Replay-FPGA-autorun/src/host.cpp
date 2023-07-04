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


// queue properties to enable profiling
property_list prop_list { property::queue::enable_profiling() };

template <typename Tin, typename Tout>
void DoWorkMultiKernel(sycl::queue& q, Tin* in, int* sampled_idx, Tout* out_pr_sampled, Tout* out_pr_insertion,
                        fixed_root &root_pr, 
                        size_t chunks, size_t chunk_size, size_t total_count,
                        size_t inflight_kernels, size_t iterations);

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
fpga_tools::Autorun<Itm1> ar_kernel1{selector, MyAutorun_itm<Itm1, fixed_l1, fixed_root, ProducePipe, L1_L2_Pipe, 1, Lev1_Width>{}};
fpga_tools::Autorun<Itm2> ar_kernel2{selector, MyAutorun_itm<Itm2, fixed_l2, fixed_l1, L1_L2_Pipe, L2_L3_Pipe, 2, Lev2_Width>{}};
fpga_tools::Autorun<Itm3> ar_kernel3{selector, MyAutorun_lastlev<Itm3, fixed_l3, fixed_l2, L2_L3_Pipe, ConsumePipe1, ConsumePipe2, ConsumePipe3, Lev3_Width>{}};

int main(int argc, char* argv[]) {
  // default values
  #if defined(FPGA_EMULATOR)
    size_t chunks = 1 << 4;         // 16
    size_t chunk_size = 1;    // 1
    size_t iterations = 1;
  #elif defined(FPGA_SIMULATOR)
    size_t chunks = 1 << 4;         // 16
    size_t chunk_size = 1;    // 1
    size_t iterations = 1;
  #else
    size_t chunks = 1 << 4;         // 16
    size_t chunk_size = 1;   // 1
    size_t iterations = 1;
  #endif

  // This is the number of kernels we will have in the queue at a single time.
  // If this number is set too low (e.g. 1) then we don't take advantage of
  // fast kernel relaunch (see the README). If this number is set to high,
  // then the first kernel launched finishes before we are done launching all
  // the kernels and therefore throughput is decreased.
  size_t inflight_kernels = 8;
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
    sibit_io* in;
    int* out_sampled_idx; //sampling output
    fixed_l3* out_sampled_value; //sampling output
    fixed_l3* out_insertion_getPr_value; //insertion output
    if ((in = malloc_host<sibit_io>(total_count*4, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
    if ((out_sampled_idx = malloc_host<int>(total_count*4, q)) == nullptr
        || (out_sampled_value = malloc_host<fixed_l3>(total_count*4, q)) == nullptr
        || (out_insertion_getPr_value = malloc_host<fixed_l3>(total_count*4, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for out arrs\n";
      std::terminate();
    }
   
    // ========== TestBench: Init ===========
    fixed_root root_pr=0;
    // Init_Tree(q,root_pr);
    
    // ========== TestBench: Insertion - get priority ===========
    // generate the input data for insertion
    // get_pr_value
    for (size_t ii=0; ii<64; ii++){
      in[ii].sampling_flag=0;
      in[ii].update_flag=0;
      in[ii].get_priority_flag=1;
      in[ii].init_flag=0;
      in[ii].pr_idx=ii;
    }

    std::cout << "Running the get-priority kernel\n";
    DoWorkMultiKernel<sibit_io,fixed_l3>(q, in, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
    root_pr, 64, chunk_size, 64, inflight_kernels, iterations);
    // validate the results 
    printf("out_pr_insertion[ii]: ");
    for (size_t ii=0; ii<64; ii++){
      printf("%f ", out_insertion_getPr_value[ii]); //should be all 0 if static init is successful. 
    }    
    std::cout << "\n";

    // ========== TestBench: Insertion - Update ===========
    for (int ii=0; ii<64; ii++){
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
    std::cout << "Running the update kernel\n";
    DoWorkMultiKernel<sibit_io,fixed_l3>(q, in, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
    root_pr, 64, chunk_size, 64, inflight_kernels, iterations);
    

    // validate the results 
    std::cout << "Completed the update kernel\n";
    std::cout <<"Root value (updated): "<< root_pr << "\n";//should return 6.4.
    // On the FPGA side: PRINTF in the producer (lev1) should accumulates to 1.6 in the end. 

    // ========== TestBench: Sampling ===========
    // size should be chunks
    fixed_root tb_rand[16]={0.1, 1.0, 1.4, 6.3, 5.0, 3.1, 3.7, 2.0, 6.1, 0.2, 0.9, 1.7, 3.3, 2.8, 4.1, 3.2};
    for (size_t ii=0; ii<chunks; ii++){
      in[ii].sampling_flag=1;
      in[ii].update_flag=0;
      in[ii].get_priority_flag=0;
      in[ii].init_flag=0;
      in[ii].start=0;
      in[ii].newx=tb_rand[ii];
    }
    std::cout << "Running the Sampling kernel\n";
    DoWorkMultiKernel<sibit_io,fixed_l3>(q, in, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
    root_pr, chunks, chunk_size, total_count, inflight_kernels, iterations);

    // validate the results 
    std::cout << "Sampled indices results:\n";
    for (size_t ii=0; ii<64; ii++){
      printf("%d ", out_sampled_idx[ii]);
    } 
    // 1st 16 elements should be: {0 9 13 62 49 30 36 19 60 1 8 16 32 27 40 31}.
    std::cout << "Sampled values results:\n";
    for (size_t ii=0; ii<64; ii++){
      printf("%f ", out_sampled_value[ii]);
    } 
    // 1st 16 elements should all be 0.1. 
    ////////////////////////////////////////////////////////////////////////////

    // free the USM pointers
    sycl::free(in, q);
    sycl::free(out_sampled_idx, q);
    sycl::free(out_sampled_value, q);
    sycl::free(out_insertion_getPr_value, q);

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
void DoWorkMultiKernel(sycl::queue& q, Tin* in, int* sampled_idx, Tout* out_pr_sampled, Tout* out_pr_insertion,
                        fixed_root &root_pr, 
                        size_t chunks, size_t chunk_size, size_t total_count,
                        size_t inflight_kernels, size_t iterations) {
  // timing data
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);
  
  size_t in_chunk = 0; // count the number of chunks for which kernels have been started
  size_t out_chunk = 0; // count the number of chunks for which kernels have finished 
  // use a queue to track the kernels in flight
  std::queue<std::pair<event,event>> event_q;
  for (size_t i = 0; i < iterations; i++) {
    // reset the output data to catch any untouched data
    std::fill_n(sampled_idx, total_count, -1);
    std::fill_n(out_pr_sampled, total_count, -1);
    std::fill_n(out_pr_insertion, total_count, -1);
    // reset counters
    in_chunk = 0;
    out_chunk = 0;
    // clear the queue
    std::queue<std::pair<event,event>> clear_q;
    std::swap(event_q, clear_q);
    // latency timers
    high_resolution_clock::time_point first_data_in, first_data_out;
    // launch the worker kernels (rmm.hpp)
    // NOTE: these kernels will process ALL of the data (total_count)
    // while the producer/consumer will be broken into chunks
    // auto events = Submit_Intermediate_SiblingItr<Itm1,fixed_l2,fixed_l3,L1_L2_Pipe,L2_L3_Pipe,2,16>(q, total_count);
    // auto events = SubmitMultiKernelWorkers<T,
    //                                        ProducePipe,
    //                                        ConsumePipe>(q, total_count);
    auto start = high_resolution_clock::now();

    do {
      // if we still have kernels to launch, launch them in here
      if (in_chunk < chunks) {
        //===perform update on root if necessary
        // std::cout<<"in_chunk: "<<in_chunk<<"\n";
        if (in[in_chunk].update_flag==1){
          root_pr+=in[in_chunk].update_offset_array[0];
        }
        
        // launch the producer/consumer pair for the next chunk of data
        size_t chunk_offset = in_chunk*chunk_size;
        // std::cout<<"chunk_offset: "<<chunk_offset<<"\n";

        // these functions are defined in 'multi_kernel.hpp'
        event p_e = Submit_Producer<ProducePipe>(q, in + chunk_offset, chunk_size);   
        event c_e = Submit_Consumer<ConsumePipe1, ConsumePipe2, ConsumePipe3, fixed_l3>(q, 
                                                   sampled_idx + chunk_offset,
                                                   out_pr_sampled + chunk_offset,
                                                   out_pr_insertion + chunk_offset,
                                                   chunk_size);

        // push the kernel event into the queue
        event_q.push(std::make_pair(p_e, c_e));

        // if this is the first chunk, track the time
        if (in_chunk == 0) first_data_in = high_resolution_clock::now();
        in_chunk++;
      }

      // wait on the oldest kernel to finish if any of these conditions are met:
      //    1) there are a certain number kernels in flight
      //    2) all of the kernels have been launched
      //
      // NOTE: 'inflight_kernels' is now the number of inflight
      // producer/consumer kernel pairs
      if ((event_q.size() >= inflight_kernels) || (in_chunk >= chunks)) {
        // grab the oldest kernel event we are waiting on
        auto event_pair = event_q.front();
        event_q.pop();

        // wait on the producer/consumer kernel pair to finish
        event_pair.first.wait();    // producer
        event_pair.second.wait();   // consumer

        // track the time if this is the first producer/consumer pair
        if (out_chunk == 0) first_data_out = high_resolution_clock::now();

        // at this point the first 'out_chunk' chunks are ready to be
        // processed on the host
        out_chunk++;
      }
    } while(out_chunk < chunks);

    // wait for the worker kernels to finish, which should be done quickly
    // since all producer/consumer pairs are done
    // for (auto& e : events) {
    //   e.wait();
    // }


    auto end = high_resolution_clock::now();

    // compute latency and processing time
    duration<double, std::milli> latency = first_data_out - first_data_in;
    duration<double, std::milli> process_time = end - start;
    latency_ms[i] = latency.count();
    process_time_ms[i] = process_time.count();
  }

  // compute and print timing information
  // PrintPerformanceInfo<T>("Multi-kernel",total_count, latency_ms, process_time_ms);
}

void Init_Tree(sycl::queue& q, fixed_root &root_pr){

    int chunk_size = 2; 
    size_t chunks = 1;       

    std::cout << "hi from init tree\n";
    sibit_io *in;
    int *sampled_idx;
    fixed_l3 *out_pr_sampled;
    fixed_l3 *out_pr_insertion;
    if ((sampled_idx = malloc_host<int>(2, q)) == nullptr ||
        (out_pr_sampled = malloc_host<fixed_l3>(2, q)) == nullptr ||
        (out_pr_insertion = malloc_host<fixed_l3>(2, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out'\n";
      std::terminate();
    }
    if ((in = malloc_host<sibit_io>(2, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
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
    std::cout << "hi\n";

    std::queue<std::pair<event,event>> event_q;
    // reset counters
    size_t in_chunk = 0;
    size_t out_chunk = 0;

    // clear the queue
    std::queue<std::pair<event,event>> clear_q;
    std::swap(event_q, clear_q);
    // auto events = Submit_Intermediate_SiblingItr<Itm1,fixed_l2,fixed_l3,L1_L2_Pipe,L2_L3_Pipe,2,16>(q, chunk_size*chunks);
    while(out_chunk < chunks){
      
      // if we still have kernels to launch, launch them in here
      if (in_chunk < chunks) {
        // launch the producer/consumer pair for the next chunk of data
        size_t chunk_offset = in_chunk*chunk_size;
        // event p_e = Submit_Producer_SiblingItr<L1_L2_Pipe>(q, in + chunk_offset, 0,chunk_size);
        // event c_e = Submit_Consumer_SiblingItr<fixed_l3, fixed_l2, L2_L3_Pipe, Lev3_Width>(q, sampled_idx + chunk_offset,
        //                                            out_pr_sampled + chunk_offset,
        //                                            out_pr_insertion + chunk_offset,
        //                                            chunk_size);
        event p_e = Submit_Producer<ProducePipe>(q, in + chunk_offset, chunk_size);   
        event c_e = Submit_Consumer<ConsumePipe1, ConsumePipe2, ConsumePipe3, fixed_l3>(q, 
                                                   sampled_idx + chunk_offset,
                                                   out_pr_sampled + chunk_offset,
                                                   out_pr_insertion + chunk_offset,
                                                   chunk_size);        
        // push the kernel event into the queue
        event_q.push(std::make_pair(p_e, c_e));
        // if this is the first chunk, track the time
        // if (in_chunk == 0) first_data_in = high_resolution_clock::now();
        in_chunk++;
      }
      if ((in_chunk >= chunks)) {
        // grab the oldest kernel event we are waiting on
        auto event_pair = event_q.front();
        event_q.pop();
        // wait on the producer/consumer kernel pair to finish
        event_pair.first.wait();    // producer
        event_pair.second.wait();   // consumer
        // track the time if this is the first producer/consumer pair
        // if (out_chunk == 0) first_data_out = high_resolution_clock::now();
        // at this point the first 'out_chunk' chunks are ready to be processed on the host
        out_chunk++;
      }
    }
    // events.wait();
    std::cout<<"out_pr_insertion for get priority:"<<out_pr_insertion[0]<<" "<<out_pr_insertion[1]<<"\n";
    // should print 0 0
    sycl::free(in, q);
    sycl::free(out_pr_insertion, q);
    sycl::free(out_pr_sampled,q);
    sycl::free(sampled_idx,q);

    printf("Init tree done. \n\n\n");
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

