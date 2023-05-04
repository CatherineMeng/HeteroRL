#ifndef __RMM_HPP__
#define __RMM_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iomanip>  // for std::setprecision
#include <iostream>
#include <vector>

// #include "exception_handler.hpp"
#include "include/exception_handler.hpp"

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

using l1_out_pipe = ext::intel::pipe< // Defined in the SYCL headers.
    class L1PipeId,                // An identifier for the pipe. can be any name
    sibit_io,                                         // The type of data in the pipe.
    8>;                                          // The capacity of the pipe.

using l2_out_pipe = ext::intel::pipe< // Defined in the SYCL headers.
    class L2PipeId,                // An identifier for the pipe.
    sibit_io,                                         // The type of data in the pipe.
    8>; 

class P;
class C;
class Itm1;

// e.g. (root is lev 0, lev 1 is the host->fpga producer module) 
// If level 2, TLevDType = fixed_l2 (priority data type stored on-chips), ParTLevDType = fixed_l1
// int treelayer=2. layersize is the total width of the tree layer.
template<typename KernelClass, typename TLevDType, typename ParTLevDType, typename InPipe, typename OutPipe, int treelayer, int layersize>
event Submit_Intermediate_SiblingItr(sycl::queue &q, size_t batch_size) {
    return q.single_task<KernelClass>([=]() [[intel::kernel_args_restrict]] {
      // Declare the SRAM array for the current trere layer
        // [[intel::singlepump,
        // intel::fpga_memory("MLAB"),
        // intel::numbanks(1)]]
        // static TLevDType TLev[layersize];
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        TLevDType TLev[layersize];

        for (size_t j = 0; j < batch_size; j++) {
            sibit_io data_in = InPipe::read();
            sibit_io data_out;
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
                    PRINTF("sibiter l2: in for loop i=%d, local_pref=%f\n",i,float(local_pref));
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
            OutPipe::write(data_out); 
            //The update operation does not change data to be sent to the next stage, so can write out before doing update
            if (data_in.update_flag==1){
                // ======== Update in a Sibling iterator ========
                int idx = data_in.update_index_array[treelayer];
                TLevDType val = data_in.update_offset_array[treelayer];
                TLev[idx]+=val;
            }
        }
    });
}

// CPU->FPGA
// assume input data fprmat is already handled at host (in_ptr is an array of sibit_io objects)
// This is always for level 1, TLevDType = fixed_l1, layersize is the tree fanouts.
template<typename OutPipe>
event Submit_Producer_SiblingItr(sycl::queue &q, sibit_io* in_ptr, fixed_root x, size_t chunk_size) {
    return q.single_task<P>([=]() [[intel::kernel_args_restrict]] {
        // Declare the SRAM array for the current trere layer
        // [[intel::singlepump,
        // intel::fpga_memory("MLAB"),
        // intel::numbanks(1)]]
        // static fixed_l1 TLev[K];
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        fixed_l1 TLev[K];
        host_ptr<sibit_io> in(in_ptr);
        for (size_t j = 0; j < chunk_size; j++) {
            sibit_io data_in = in[j];
            sibit_io data_out;
            if (data_in.init_flag==1){
                for (size_t k=0; k<K; k++){
                    TLev[k]=0;
                }
            }
            if (data_in.sampling_flag==1){
                // ======== Sampling in a Sibling iterator ========
                fixed_root local_pref=0;
                fixed_root prev_pref=0;
                #pragma pipeline
                for (int i=0; i<K; i++){
                    PRINTF("sibiter l1: in for loop i=%d, local_pref=%f\n",i,float(local_pref));
                    prev_pref=local_pref;
                    local_pref+=TLev[i];
                    if (local_pref >= x){
                        //calculating the start addr of next layer sibling iteration
                        data_out.start=(i)*K;
                        //calculating the new x to be found in the next layer
                        data_out.newx=x-prev_pref;
                        OutPipe::write(data_out); 
                        break;		
                    }
                }
            }
            //The update operation does not change data to be sent to the next stage, so can write out before doing update
            if (data_in.update_flag==1){
                // ======== Update in a Sibling iterator ========
                int idx = data_in.update_index_array[1];
                fixed_l1 val = data_in.update_offset_array[1];
                TLev[idx]+=val;
            }
            // if (data_in.get_priority_flag==1){
                // ======== pass the idx down for obtaining TD at the leaf level (Consumer kernel) ========
            // }
        }
        // Used for testbench, remove when test passed
        PRINTF("Lev 1 TLev[0]: %f\n",TLev[0]);
    });
}

// FPGA->CPU
// If level D-1=3 is last level, LastLevDType = fixed_l3, ParTLevDType = fixed_l2
template<typename LastLevDType, typename ParTLevDType, typename InPipe, int layersize>
event Submit_Consumer_SiblingItr(sycl::queue &q, int* out_ptr_sampled_idx, 
LastLevDType* out_ptr_sampled_value, LastLevDType* out_ptr_getPr_value, size_t chunk_size) {
    return q.single_task<C>([=]() [[intel::kernel_args_restrict]] {
        // [[intel::singlepump,
        // intel::fpga_memory("MLAB"),
        // intel::numbanks(1)]]
        // static LastLevDType TLev[layersize];
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        LastLevDType TLev[layersize];

        host_ptr<int> out_sampled_idx(out_ptr_sampled_idx);
        host_ptr<LastLevDType> out_sampled_value(out_ptr_sampled_value);
        host_ptr<LastLevDType> out_getPr_value(out_ptr_getPr_value);
        for (size_t j = 0; j < chunk_size; j++) {
            sibit_io data_in=InPipe::read();
            sibit_io data_out;
            if (data_in.init_flag==1){
                PRINTF("Doing INIT last tree level.\n");
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
                    PRINTF("sibiter l3: in for loop i=%d, local_pref=%f\n",i,float(local_pref));
                    if (local_pref>=x){
                        break;
                    }
                    prev_pref=local_pref;
                    local_pref+=TLev[i];
                }
                i=i-1;
                out_sampled_idx[j]=i;
                out_sampled_value[j]=TLev[i];
            }
            if (data_in.update_flag==1){
                // ======== Update in a Sibling iterator ========
                int idx = data_in.update_index_array[D-1];
                LastLevDType val = data_in.update_offset_array[D-1];
                TLev[idx]+=val;
            }
            if (data_in.get_priority_flag==1){
                // ======== Last Level, Get Priority value for Insertion ========
                int tidx=data_in.pr_idx;
                out_getPr_value[j]=TLev[tidx];
            }
        }
    });
}


#endif /* __RMM_HPP__ */