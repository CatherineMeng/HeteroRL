#ifndef __RMM_HPP__
#define __RMM_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iomanip>  // for std::setprecision
#include <iostream>
#include <vector>

// #include "exception_handler.hpp"
#include "include/exception_handler.hpp"

#include "autorun.hpp"

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;
// using namespace std;

#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }
  
//#define K 128 //fanout 
//#define Lev1_Width 79
//#define Lev2_Width 10000 //78*128+16
#define K 4 //fanout 
#define D 4 //depth with root
#define Lev1_Width 4
#define Lev2_Width 16
#define Lev3_Width 64

using fixed_root = float;
using fixed_l1 = float;
using fixed_l2 = float;
using fixed_l3 = float;

using fixed_upd = float;
using fixed_insrt = float;

using fixed_bool = bool;

// data pack transfered along sibling iterators
// update_offsets are calculated on the host based on the difference between sampled value and value to be updated
// insertion is performed by get_priority followed by update from host (1. get_priority to get the old TD value, 2. update new TD value)
typedef struct {
	// ---sampling---
	fixed_bool sampling_flag;
	int start; //the start index of next layer sibling iteration
	fixed_root newx; //the new prefix sum value to be found in the next layer
	// ---update---
	fixed_bool update_flag;
	int update_index_array[D]; //root lev index =0, so [0, ...]
	fixed_root update_offset_array[D]; //root lev index =0
	// ---get_priority---
	fixed_bool get_priority_flag;
	int pr_idx; //the index used at the last tree level for obtaining old TD
    // ---tree initializaation (do once)---
    fixed_bool init_flag;
} sibit_io;

class P;
class C;
class Itm1;
class Itm2;
class Itm3;

// DONE: make it autorun
// e.g. (root is lev 0, lev 1 is the host->fpga producer module) 
// If level 2, TLevDType = fixed_l2 (priority data type stored on-chips), ParTLevDType = fixed_l1
// int treelayer=2. layersize is the total width of the tree layer.

// template<typename KernelClass, typename TLevDType, typename ParTLevDType, typename InPipe, typename OutPipe, int treelayer, int layersize>
// event Submit_Intermediate_SiblingItr(sycl::queue &q, size_t batch_size) {
template<typename KernelClass, typename TLevDType, typename ParTLevDType, typename InPipe, 
         typename OutPipe, int treelayer, int layersize>
struct MyAutorun_itm {
    // printf( "submit Intermediate_SiblingItr\n");
    // return q.single_task<KernelClass>([=]() [[intel::kernel_args_restrict]] {
    void operator()() const {
      // Declare the SRAM array for the current trere layer
        // [[intel::singlepump,
        // intel::fpga_memory("MLAB"),
        // intel::numbanks(1)]]
        // static TLevDType TLev[layersize];
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        TLevDType TLev[layersize];
        // initialize on-chip memory
        for (size_t j=0;j<layersize;j++){
            TLev[j]=0;
        }
        // PRINTF("itm single_task\n");
        // for (size_t j = 0; j < batch_size; j++) {
        while(1){
            // PRINTF("itm kernel, j: %d\n",j);
            sibit_io data_in = InPipe::read();
            sibit_io data_out = data_in; //make a copy of data_in
            if (data_in.init_flag==1){
                for (size_t k=0; k<layersize; k++){
                    TLev[k]=0;
                }
            }
            if (data_in.sampling_flag==1){
                // ======== Sampling in a Sibling iterator ========
                ParTLevDType local_pref=0;
                ParTLevDType prev_pref=0;
                int pipstart=data_in.start;
                int pipbound=data_in.start+K;
                fixed_root x=data_in.newx;
                int i=pipstart;
                #pragma pipeline
                for (i=pipstart; i<pipbound; i++){
                    TLevDType tmpload = TLev[i];
                    // PRINTF("sibiter l2: in for loop i=%d, local_pref=%f\n",i,float(local_pref));
                    if (local_pref>=x){
                        break;
                    }
                    prev_pref=local_pref;
                    local_pref+=tmpload;
                }
                i = i -1;
                //calculating the start addr of next layer sibling iteration
                data_out.start=(i)*K;
                //calculating the new x to be found in the next layer
                data_out.newx=x-prev_pref;
            }
            //the get_prioriy of non-last-level kernels simply pass the input data to the next stage
            OutPipe::write(data_out); 
            //The update operation does not change data to be sent to the next stage, so can write out before doing update
            if (data_in.update_flag==1){
                // ======== Update in a Sibling iterator ========
                int idx = data_in.update_index_array[treelayer];
                TLevDType val = data_in.update_offset_array[treelayer];
                TLev[idx]+=val;
            }
        }
    // });
    }
// }
};

// DONE: make it autorun
// FPGA->CPU
// If level D-1=3 is last level, LastLevDType = fixed_l3, ParTLevDType = fixed_l2
// Outpipe1 is for sampled_idx (int)
// Outpipe2 is for sampled_value (fixed_l3, or LastLevDType)
// Outpipe3 is for get-prioriy-value (fixed_l3, or LastLevDType)
// template<typename KernelClass, typename LastLevDType, typename ParTLevDType, 
//          typename InPipe, typename OutPipe1, typename OutPipe2, typename OutPipe3, int layersize>
// event Submit_LastLevel_SiblingItr(sycl::queue &q, size_t chunk_size) {
template<typename KernelClass, typename LastLevDType, typename ParTLevDType, 
         typename InPipe, typename OutPipe1, typename OutPipe2, typename OutPipe3, int layersize>
struct MyAutorun_lastlev {
    // printf("submit the consumer;\n");
    // return q.single_task<C>([=]() [[intel::kernel_args_restrict]] {
    void operator()() const {
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        LastLevDType TLev[layersize];
        // PRINTF("consumer, chunk_size: %d\n",chunk_size);
        // Initialize on-chip memory
        for (size_t j=0;j<layersize;j++){
            TLev[j]=0;
        }
        while(1) {
            sibit_io data_in=InPipe::read();
            int data_out_1 =0; //sampled idx
            LastLevDType data_out_2=0; //sampled value
            LastLevDType data_out_3=0; //get-priority value
            if (data_in.init_flag==1){
                // PRINTF("Doing INIT last tree level.\n");
                for (size_t k=0; k<layersize; k++){
                    TLev[k]=0;
                }
            }
            if (data_in.sampling_flag==1){
                // ======== Sampling in a Sibling iterator ========
                ParTLevDType local_pref=0;
                ParTLevDType prev_pref=0;
                int pipstart=data_in.start;
                int pipbound=data_in.start+K;
                fixed_root x=data_in.newx;
                int i=pipstart;
                #pragma pipeline
                for (i=pipstart; i<pipbound; i++){
                    // PRINTF("sibiter l3: in for loop i=%d, local_pref=%f\n",i,float(local_pref));
                    // printf("sibiter l3: in for loop i=%d, local_pref=%f\n",i,float(local_pref));
                    if (local_pref>=x){
                        break;
                    }
                    prev_pref=local_pref;
                    local_pref+=TLev[i];
                }
                i=i-1;
                data_out_1=i;
                data_out_2=TLev[i];
            }
            // out_sampled_idx[j]=i;
            OutPipe1::write(data_out_1);
            // out_sampled_value[j]=TLev[i];
            OutPipe2::write(data_out_2);
            if (data_in.update_flag==1){
                // ======== Update in a Sibling iterator ========
                int idx = data_in.update_index_array[D-1];
                LastLevDType val = data_in.update_offset_array[D-1];
                TLev[idx]+=val;
            }
            if (data_in.get_priority_flag==1){
                // ======== Last Level, Get Priority value for Insertion ========
                int tidx=data_in.pr_idx;
                data_out_3=TLev[tidx];
                // out_getPr_value[j]=TLev[tidx];
            }
            OutPipe3::write(data_out_3);
        }
    // });
    }
// }
};

// template<typename InPipe, typename OutPipe1, typename OutPipe2, typename OutPipe3>
// std::vector<event> SubmitMultiKernelWorkers_AR(queue &q, size_t size) {
//   // internal pipes between kernels
//   using L1_L2_Pipe = ext::intel::pipe<class L1_L2_PipeClass, sibit_io,16>;
//   using L2_L3_Pipe = ext::intel::pipe<class L2_L3_PipeClass, sibit_io,16>;

//   // submit the kernels
//   event e0 = Submit_Intermediate_SiblingItr<Itm1, fixed_l1, fixed_root, InPipe, L1_L2_Pipe, 1, Lev1_Width>(q, size);
//   event e1 = Submit_Intermediate_SiblingItr<Itm2, fixed_l2, fixed_l1, L1_L2_Pipe, L2_L3_Pipe, 2, Lev2_Width>(q, size);
//   event e2 = Submit_LastLevel_SiblingItr<Itm3, fixed_l3, fixed_l2, L2_L3_Pipe, OutPipe1, OutPipe2, OutPipe3, Lev3_Width>(q, size);

//   // return the events
//   return {e0, e1, e2};
// }

// template<typename InPipe, typename OutPipe1, typename OutPipe2, typename OutPipe3>
// void SubmitMultiKernelWorkers_AR() {
//   // internal pipes between kernels
//   using L1_L2_Pipe = ext::intel::pipe<class L1_L2_PipeClass, sibit_io,16>;
//   using L2_L3_Pipe = ext::intel::pipe<class L2_L3_PipeClass, sibit_io,16>;

//   // submit the kernels
// //   event e0 = Submit_Intermediate_SiblingItr<Itm1, fixed_l1, fixed_root, InPipe, L1_L2_Pipe, 1, Lev1_Width>(q, size);
// //   event e1 = Submit_Intermediate_SiblingItr<Itm2, fixed_l2, fixed_l1, L1_L2_Pipe, L2_L3_Pipe, 2, Lev2_Width>(q, size);
// //   event e2 = Submit_LastLevel_SiblingItr<Itm3, fixed_l3, fixed_l2, L2_L3_Pipe, OutPipe1, OutPipe2, OutPipe3, Lev3_Width>(q, size);

//   // declaring a global instance of this class causes the constructor to be called
//   // before main() starts, and the constructor launches the kernel.
//   fpga_tools::Autorun<Itm1> ar_kernel1{selector, MyAutorun_itm<Itm1, fixed_l1, fixed_root, InPipe, L1_L2_Pipe, 1, Lev1_Width>{}};
//   fpga_tools::Autorun<Itm2> ar_kernel2{selector, MyAutorun_itm<Itm2, fixed_l2, fixed_l1, L1_L2_Pipe, L2_L3_Pipe, 2, Lev2_Width>{}};
//   fpga_tools::Autorun<Itm3> ar_kernel3{selector, MyAutorun_itm<Itm3, fixed_l3, fixed_l2, L2_L3_Pipe, OutPipe1, OutPipe2, OutPipe3, Lev3_Width>{}};

// }



// CPU->FPGA
// assume input data fprmat is already handled at host (in_ptr is an array of sibit_io objects)
// This is always for level 1, TLevDType = fixed_l1, layersize is the tree fanouts.
template<typename OutPipe>
event Submit_Producer(queue &q, sibit_io *in_buf, size_t size) {
    // printf( "submit the producer\n");
    return q.single_task<P>([=]() [[intel::kernel_args_restrict]] {
        host_ptr<sibit_io> in(in_buf);
        // int size = in_buf.size();
        for (size_t j = 0; j < size; j++) {
            // PRINTF("producer, itr j=%d\n",j);
            OutPipe::write(in[j]); 
        }        
    });
}

// Inpipe1 is for sampled_idx (int)
// Inpipe2 is for sampled_value (fixed_l3, or LastLevDType)
// Inpipe3 is for get-prioriy-value (fixed_l3, or LastLevDType)
// template <typename InPipe1, typename InPipe2, typename InPipe3, typename LastLevDType>
// event Submit_Consumer(queue& q, buffer<int, 1>& out_buf1, 
//                            buffer<LastLevDType, 1>& out_buf2, buffer<LastLevDType, 1>& out_buf3) {
//   return q.submit([&](handler& h) {
//     accessor out1(out_buf1, h, write_only, no_init);
//     accessor out2(out_buf2, h, write_only, no_init);
//     accessor out3(out_buf3, h, write_only, no_init);
//     int size = out_buf1.size();
//     h.single_task<KernelID>([=] {
//       for (int i = 0; i < size; i++) {
//         out1[i] = InPipe1::read();
//         out2[i] = InPipe2::read();
//         out3[i] = InPipe3::read();
//       }
//     });
//   });
// }
template <typename InPipe1, typename InPipe2, typename InPipe3, typename LastLevDType>
event Submit_Consumer(queue& q, int *out_buf1, LastLevDType *out_buf2, LastLevDType *out_buf3, size_t size) {
  return q.single_task<C>([=]() [[intel::kernel_args_restrict]] {
    host_ptr<int> out1(out_buf1);
    host_ptr<LastLevDType> out2(out_buf2);
    host_ptr<LastLevDType> out3(out_buf3);
    // int size = out_buf1.size();
    for (int i = 0; i < size; i++) {
        auto data1 = InPipe1::read();
        *(out1 + i) = data1;
        auto data2 = InPipe2::read();
        *(out2 + i) = data2;
        auto data3 = InPipe3::read();
        *(out3 + i) = data3;
    }
  });
}


#endif /* __RMM_HPP__ */