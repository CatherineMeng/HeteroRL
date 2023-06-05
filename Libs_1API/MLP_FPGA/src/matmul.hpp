#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

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
  
//MV: 1,LL * LL,NL
#define LL 16 //input dimension
#define NL 256 //output dimension
#define NL_chunksize 32 //output dimension


using weight_fmt = float;
using act_fmt = float;


// data chunk whose size determins the parallelization factor along the output dimension
// typedef struct {
// 	weight_fmt dat[NL_chunksize];
// } blockvec;

class P;
class C;
class MM;


template<typename KernelClass, typename InPipe, typename OutPipe, int LL_nn, int NL_nn>
struct MyAutorun_MM {

    void operator()() const {

        [[intel::singlepump,
        intel::fpga_memory("BLOCK_RAM"),
        intel::numbanks(NL_nn),
        intel::max_replicates(NL_nn)]]
        weight_fmt W[LL_nn][NL_nn];
        // initialize on-chip memory
        for (size_t i=0;i<LL_nn;i++){
            for (size_t j=0;j<NL_nn;j++)
            W[i][j]=(i+j)/10;
        }
        
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        act_fmt In[LL_nn];
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        act_fmt Out[NL_nn];
        for (size_t i=0; i<NL_nn; i++)Out[i]=0;

        PRINTF("in fpga kernel ...\n");

        while(1){
            // PRINTF("itm kernel, j: %d\n",j);
            for (size_t i=0; i<LL_nn; i++)In[i] = InPipe::read();
            for (size_t i=0; i<LL_nn; i++){
                #pragma unroll
                for (size_t j=0; j<NL_nn; j++){
                    Out[j]+=In[i]*W[i][j];
                }
            }
            for (size_t i=0; i<NL_nn; i++)OutPipe::write(Out[i]); 
        }

    }

};

// CPU->FPGA
// assume input data fprmat is already handled at host (in_ptr is an array of act_fmt objects)
// This is always for level 1, TLevDType = fixed_l1, layersize is the tree fanouts.
template<typename OutPipe>
event Submit_Producer(queue &q, act_fmt *in_buf, size_t size) {
    // printf( "submit the producer\n");
    return q.single_task<P>([=]() [[intel::kernel_args_restrict]] {
        host_ptr<act_fmt> in(in_buf);
        for (size_t j = 0; j < size; j++) {
            OutPipe::write(in[j]); 
        }        
    });
}


template <typename InPipe>
event Submit_Consumer(queue& q, act_fmt *out_buf, size_t size) {
  return q.single_task<C>([=]() [[intel::kernel_args_restrict]] {
    host_ptr<act_fmt> out(out_buf);
    for (size_t i = 0; i < size; i++) {
        auto data = InPipe::read();
        *(out + i) = data;
    }
  });
}


#endif /* __MATMUL_HPP__ */