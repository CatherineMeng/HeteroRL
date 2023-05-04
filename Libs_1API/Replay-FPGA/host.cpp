//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

#include "rmm.hpp"

using namespace sycl;
using namespace std::chrono;

#include "include/exception_handler.hpp"


template <typename Tin, typename Tout>
void DoWorkMultiKernel(queue& q, Tin* in, int* sampled_idx, Tout* out_pr_sampled, Tout* out_pr_insertion, 
                        size_t chunks, size_t chunk_size, size_t total_count,
                        size_t inflight_kernels, size_t iterations);

template<typename T>
void PrintPerformanceInfo(std::string print_prefix, size_t count,
                          std::vector<double>& latency_ms,
                          std::vector<double>& process_time_ms);


int main(int argc, char* argv[]) {
  // default values
  #if defined(FPGA_EMULATOR)
    size_t chunks = 1 << 4;         // 16
    size_t chunk_size = 1;    // 1
    size_t iterations = 2;
  #elif defined(FPGA_SIMULATOR)
    size_t chunks = 1 << 4;         // 16
    size_t chunk_size = 1;    // 1
    size_t iterations = 2;
  #else
    size_t chunks = 1 << 4;         // 16
    size_t chunk_size = 1;   // 1
    size_t iterations = 2;
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
  std::cout << "Chunk count:          " << chunk_size << "\n";
  std::cout << "Total count:          " << total_count << "\n";
  std::cout << "Iterations:           " << iterations-1 << "\n";
  std::cout << "\n";

  bool passed = true;

  try {
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

    // create the device queue
    queue q(selector, fpga_tools::exception_handler, prop_list);

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
    if ((in = malloc_host<sibit_io>(total_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
    if ((out_sampled_idx = malloc_host<int>(total_count, q)) == nullptr
        || (out_sampled_value = malloc_host<fixed_l3>(total_count, q)) == nullptr
        || (out_insertion_getPr_value = malloc_host<fixed_l3>(total_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for out arrs\n";
      std::terminate();
    }

    // generate the input data
    // NOTE: by generating all of the data ahead of time, we are essentially
    // assuming that the producer of data (producing data for the FPGA to
    // consume) has infinite bandwidth. However, if the producer of data cannot
    // produce data faster than our FPGA can consume it, the CPU producer will
    // bottleneck the total throughput of the design.
    std::generate_n(in, total_count, [] { return Type(rand() % 100); });

    ////////////////////////////////////////////////////////////////////////////
    // run the optimized (for latency) version with multiple kernels that uses
    // fast kernel relaunch by keeping at most 'inflight_kernels' in the SYCL
    // queue at a time
    std::cout << "Running the latency optimized multi-kernel design\n";
    DoWorkMultiKernel(q, in, out, chunks, chunk_size, total_count,
                      inflight_kernels, iterations);

    // validate the results using the lambda

    std::cout << "\n";
    ////////////////////////////////////////////////////////////////////////////

    // free the USM pointers
    sycl::free(in, q);
    sycl::free(out_sampled_idx, q);
    sycl::free(out_sampled_value, q);
    sycl::free(out_insertion_getPr_value, q);

  } catch (exception const& e) {
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



// the pipes used to produce/consume data
using ProducePipe = pipe<class ProducePipeClass, Type>;
using ConsumePipe = pipe<class ConsumePipeClass, Type>;

// Tin:sibit_io, Tout: fixed_l3
template <typename Tin, typename Tout>
void DoWorkMultiKernel(queue& q, Tin* in, int* sampled_idx, Tout* out_pr_sampled, Tout* out_pr_insertion, 
                        size_t chunks, size_t chunk_size, size_t total_count,
                        size_t inflight_kernels, size_t iterations) {
  // timing data
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);

  // count the number of chunks for which kernels have been started
  size_t in_chunk = 0;
  // count the number of chunks for which kernels have finished 
  size_t out_chunk = 0;
  // use a queue to track the kernels in flight
  std::queue<std::pair<event,event>> event_q;
  for (size_t i = 0; i < iterations; i++) {
    // reset the output data to catch any untouched data
    std::fill_n(out, total_count, -1);
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
    auto events = Submit_Intermediate_SiblingItr<Itm1,fixed_l2,fixed_l3,ProducePipe,ConsumePipe,2,16>(q, total_count);

    auto start = high_resolution_clock::now();

    do {
      // if we still have kernels to launch, launch them in here
      if (in_chunk < chunks) {
        // launch the producer/consumer pair for the next chunk of data
        size_t chunk_offset = in_chunk*chunk_size;

        fixed_root x = 0;
        // these functions are defined in 'multi_kernel.hpp'
        event p_e = SubmitProducer<T, ProducePipe>(q, in + chunk_offset, x,
                                                   chunk_size);
        event c_e = SubmitConsumer<T, ConsumePipe>(q, sampled_idx + chunk_offset,
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
    for (auto& e : events) {
      e.wait();
    }

    auto end = high_resolution_clock::now();

    // compute latency and processing time
    duration<double, std::milli> latency = first_data_out - first_data_in;
    duration<double, std::milli> process_time = end - start;
    latency_ms[i] = latency.count();
    process_time_ms[i] = process_time.count();
  }

  // compute and print timing information
  PrintPerformanceInfo<T>("Multi-kernel",
                          total_count, latency_ms, process_time_ms);
}

void Init_Tree(){
  sibit_io* in;
  in[0].sampling_flag=0;
  in[0].update_flag=0;
  in[0].get_priority_flag=0;
  in[0].init_flag=1;
  // outputs are not important, only used for SRAM tree init.
  int* sampled_idx; 
  fixed_l3* out_pr_sampled, out_pr_insertion; 
  auto events = Submit_Intermediate_SiblingItr<Itm1,fixed_l2,fixed_l3,ProducePipe,ConsumePipe,2,16>(q, total_count);
  event p_e = SubmitProducer<T, ProducePipe>(q, in, x, 1);
  event c_e = SubmitConsumer<T, ConsumePipe>(q, sampled_idx,out_pr_sampled,out_pr_insertion, 1);
  c_e.wait();

  sycl::free(in, q);
  sycl::free(sampled_idx, q);
  sycl::free(out_pr_sampled, q);
  sycl::free(out_pr_insertion, q);

  printf("Host: Initalize Tree complete.\n") 
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

