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

#include "matmul.hpp"

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

template <typename T>
void DoWorkMultiKernel(sycl::queue& q, T* in, T* out,
                        size_t chunks, size_t chunk_size, size_t total_count,
                        size_t inflight_kernels, size_t iterations);

template<typename T>
void PrintPerformanceInfo(std::string print_prefix, size_t count,
                          std::vector<double>& latency_ms,
                          std::vector<double>& process_time_ms);

// the pipes used to produce/consume data
using ProducePipe = ext::intel::pipe<class ProducePipeClass, act_fmt,16>;
using ConsumePipe = ext::intel::pipe<class ConsumePipe1Class, act_fmt,16>; //sampled ind

// internal pipes between kernels
// using L1_L2_Pipe = ext::intel::pipe<class L1_L2_PipeClass, act_fmt,16>;
// using L2_L3_Pipe = ext::intel::pipe<class L2_L3_PipeClass, act_fmt,16>;

// declaring a global instance of this class causes the constructor to be called
// before main() starts, and the constructor launches the kernel.
fpga_tools::Autorun<MM> ar_kernel1{selector, MyAutorun_MM<MM, ProducePipe, ConsumePipe, 16, 256>{}};

int main(int argc, char* argv[]) {
  // default values
  #if defined(FPGA_EMULATOR)
    size_t chunks = LL;         // 16
    size_t chunk_size = 1;    // 1
    size_t iterations = 1; //ToDo: This could be the batch size?
  #elif defined(FPGA_SIMULATOR)
    size_t chunks = LL;         // 16
    size_t chunk_size = 1;    // 1
    size_t iterations = 1; //ToDo: This could be the batch size?
  #else
    size_t chunks = LL;         // 16
    size_t chunk_size = 1;   // 1
    size_t iterations = 1; //ToDo: This could be the batch size?
  #endif

  // This is the number of kernels we will have in the queue at a single time.
  // If this number is set too low (e.g. 1) then we don't take advantage of
  // fast kernel relaunch (see the README). If this number is set to high,
  // then the first kernel launched finishes before we are done launching all
  // the kernels and therefore throughput is decreased.
  size_t inflight_kernels = LL;
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
    act_fmt* in;
    act_fmt* out; 

    if ((in = malloc_host<act_fmt>(LL, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
    if ((out = malloc_host<act_fmt>(NL, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for out\n";
      std::terminate();
    }

    //Initialize input
    for (size_t i=0; i<LL; i++) in[i]=i;
    for (size_t i=0; i<LL; i++) out[i]=0;
    // ========== TestBench ===========

    std::cout << "Running the Produce-MV-Consume kernel\n";
    // std::cout << "hi1\n";std::cout << "hi2\n";std::cout << "hi3\n";
    DoWorkMultiKernel<act_fmt>(q, in, out, chunks, chunk_size, total_count, inflight_kernels, iterations);

    // ========== validate the results ========== 
    //Initialize weight & out for software emulation comparison
    weight_fmt W_host[LL][NL];
    // initialize on-chip memory
    for (size_t i=0;i<LL;i++){
        for (size_t j=0;j<NL;j++)
        W_host[i][j]=(i+j)/10;
    }
    act_fmt* out_host;
    if ((out_host = malloc_host<act_fmt>(NL, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for out\n";
      std::terminate();
    }
    for (size_t i=0; i<NL; i++)out_host[i]=0;
    //host side MV
    for (size_t i=0; i<LL; i++){
        for (size_t j=0; j<NL; j++){
            out_host[j]+=in[i]*W_host[i][j];
        }
    }
    for (size_t ii=0; ii<8; ii++){
        if (out[ii]!=out_host[ii]){passed=false;}
    }
    std::cout << "Completed the MV kernel\n";

    // free the USM pointers
    sycl::free(in, q);
    sycl::free(out, q);
    sycl::free(out_host, q);

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
template <typename T>
void DoWorkMultiKernel(sycl::queue& q, T* in, T* out,
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
    std::cout <<"in interation i = "<<i<<"\n";
    // reset the output data to catch any untouched data
    std::fill_n(out, NL, -1);
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

        // these functions are defined in 'multi_kernel.hpp'
        event p_e = Submit_Producer<ProducePipe>(q, in + chunk_offset, chunk_size);   
        event c_e = Submit_Consumer<ConsumePipe>(q, out + chunk_offset, chunk_size);

        // push the kernel event into the queue
        event_q.push(std::make_pair(p_e, c_e));

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
      if ((event_q.size() >= inflight_kernels) || (in_chunk >= chunks)) {
        // grab the oldest kernel event we are waiting on
        auto event_pair = event_q.front();
        event_q.pop();

        // wait on the producer/consumer kernel pair to finish
        event_pair.first.wait();    // producer
        event_pair.second.wait();   // consumer

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
