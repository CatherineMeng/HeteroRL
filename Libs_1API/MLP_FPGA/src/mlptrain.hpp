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
  
//2-layer MLP, input (L1), hidden (L2), output (L3)
#define L1 16 //input dimension, size(state)
#define L2 256 //hidden dimension
#define L3 8 //output dimension, num actions
// #define NL_chunksize 32 //output dimension

// typedefs
// using weight_fmt = float; //using an array-struct for fasterr streaming
using act_fmt = float;

// Just for inf
typedef struct {
	act_fmt s[L1]; //current state
    act_fmt snt[L1]; //next state
} StateConcate;
typedef struct {
	act_fmt s[L2]; //L2 act/gr of current state
    act_fmt snt[L2]; //L2 act/gr next state
} L2ItmConcate;
typedef struct {
	act_fmt s[L3]; //L3 act/gr of current state
    act_fmt snt[L3]; //L3 act/gr next state
} L3ItmConcate;
typedef struct {
	float w[L1];
} W1Fmt; //assume weights are float
typedef struct {
	float w[L2];
} W2Fmt; //assume weights are float
//for both inf and training
typedef struct {
	float w[L3];
} W2TranspFmt; //assume weights are float
typedef struct {
	act_fmt s[L1]; //current state
} L1AG;
typedef struct {
	act_fmt s[L2]; //L2 act/gr of current state
} L2AG;
typedef struct {
	act_fmt s[L3]; //L3 act/gr of current state
} L3AG;
// juust for obj
typedef struct {
	float r; //reward
    int done; //indicate whether terminal state
} RDone;
// =========== Custom blocks based on NN and RL config ===========
// example: DQN - deepmellow (no target net))
class P; //producer
class MM_FW1;
class MM_FW2;
class OBJ;
class MM_BW;
class WA1;
class WA2;
class C; //consumer

using SinPipe = ext::intel::pipe<class SinPipeClass, StateConcate,8>; //its depth should be >= processed sub-batch size (chunk size)
using ReadW1Pipe = ext::intel::pipe<class ReadW1PipeClass, W1Fmt,L2>; //<name, datatype, pipe depth>
using ReadB1Pipe = ext::intel::pipe<class ReadB1PipeClass, float,L2>;
//sigals weight sync, 1 means yes (stream in new weights to autorun kernels), 0 means no (keep weights static)
using L1FWSigPipe = ext::intel::pipe<class L1SigPipeClass, bool,1>; 
using ReadW2Pipe = ext::intel::pipe<class ReadW2PipeClass, W2Fmt,L3>;
using ReadB2Pipe = ext::intel::pipe<class ReadB2PipeClass, float,L3>;
//sigals weight sync, 1 means yes (stream in new weights to autorun kernels), 0 means no (keep weights static)
using L2FWSigPipe = ext::intel::pipe<class L2SigPipeClass, bool,1>; 
using A0Pipe = ext::intel::pipe<class A0PipeClass, L1AG,8>; //its depth should be >= processed sub-batch size (chunk size)
// from producer directly to obj, depth should be >= processed batch size (total count)
using RDonePipe = ext::intel::pipe<class RDonePipeClass, RDone,64>; 


using L12Pipe = ext::intel::pipe<class L12PipeClass, L2ItmConcate,1>;
using A1Pipe = ext::intel::pipe<class A1PipeClass, L2AG,8>; //its depth should be >= processed sub-batch size (chunk size)

using L23Pipe = ext::intel::pipe<class L23PipeClass, L3ItmConcate,1>;

using L32Pipe = ext::intel::pipe<class L32PipeClass, L2ItmConcate,1>;
using D2Pipe = ext::intel::pipe<class D2PipeClass, L3AG,8>; //its depth should be >= processed sub-batch size (chunk size)

using D1Pipe = ext::intel::pipe<class D1PipeClass, L2AG,8>; //its depth should be >= processed sub-batch size (chunk size)

// using ConsumePipe = ext::intel::pipe<class ConsumePipe1Class, act_fmt,16>; 
using writeW1Pipe = ext::intel::pipe<class writeW1PipeClass, W1Fmt,L2>; //<name, datatype, pipe depth>
using writeB1Pipe = ext::intel::pipe<class writeB1PipeClass, float,L2>;
using writeW2Pipe = ext::intel::pipe<class writeW2PipeClass, W2Fmt,L3>;
using writeB2Pipe = ext::intel::pipe<class writeB2PipeClass, float,L3>;

// CPU->FPGA, producer for FW layers and LOSS (objctv) 
// assume input data format is already handled at host (inputs/weights/<r,done> are arrays of structs, biases are arrays of floats)
template<typename OutPipeS, typename OutPipeW1, typename OutPipeW2, typename OutPipeB1, typename OutPipeB2, typename OutSigPipeW1, typename OutSigPipeW2, typename A0bufPipe, typename rdPipe>
event Submit_Producer(queue &q, StateConcate *state_in_buf, weight_fmt1 *w1_buf, weight_fmt2 *w2_buf, act_fmt *bias1_buf, act_fmt *bias2_buf, RDone *rdone_buf,
                      size_t size, bool stream_w) { //size is (sub-)batch size here, not state size
    // std::cout<< "submit the producer\n";
    return q.single_task<P>([=]() [[intel::kernel_args_restrict]] {
        host_ptr<StateConcate> state_in(state_in_buf);
        host_ptr<weight_fmt1> w1_in(w1_buf);
        host_ptr<weight_fmt2> w2_in(w2_buf);
        host_ptr<act_fmt> b1_in(bias1_buf);
        host_ptr<act_fmt> b2_in(bias2_buf);
        host_ptr<RDone> rd_in(rdone_buf);
        for (size_t i = 0; i < size; i++) {
            OutPipeS::write(state_in[i]); 
            A0bufPipe::write(state_in[i].s);
            rdPipe::write(rd_in[i]);
        }     
        // read weights only if signaled to do so  
        if (stream_w){
            for (size_t i = 0; i < L2; i++){
                OutPipeW1::write(w1_in[i]); 
                OutPipeB1::write(b1_in[i]);
                OutSigPipeW1::write(1);
            }
            for (size_t i = 0; i < L3; i++){
                OutPipeW2::write(w2_in[i]); 
                OutPipeB2::write(b2_in[i]);
                OutSigPipeW2::write(1);
            }
        } 
        else{OutSigPipeW1::write(0);OutSigPipeW2::write(0);} 
    });
}

// CPU->FPGA, producer for BW layers 
template<typename OutPipeW, typename OutSigPipeW>
event Submit_Producer(queue &q, weight_transpose_fmt2 *w2_buf, bool stream_w) { 
    // std::cout<< "submit the producer\n";
    return q.single_task<P>([=]() [[intel::kernel_args_restrict]] {
        host_ptr<weight_transpose_fmt2> w2_in(w2_buf);
        // read weights only if signaled to do so  
        if (stream_w){
            for (size_t i = 0; i < L2; i++){
                OutPipeW::write(w2_in[i]); 
                OutSigPipeW::write(1);
            }
        } 
        else{OutSigPipeW::write(0);} 
    });
}

// LLConcateFmt: last-level input concatenation of intermediate results from current+next state
// NLConcateFmt: next-level output concatenation of intermediate results from current+next state
// NLAGFmt: next-level output intermediate results from current state only - abufPipe: pipe reserved for WA, each is NLAGFmt
// weight_fmt: concatenated as a struct in the input dimension
// InSigPipe: signals to sync weight, each is a bool
// InWPipe, InBPipe: read weights from Producer only if signaled to do so
// InPipe: read data from last layer/input, each is an LLConcateFmt
// OutPipe: write data to next layer, each is an NLConcateFmt
// abufPipe: write data to activation buffer later used for WA, each is an NLConcateFmt
// LL/NL_nn: last-level/next-level number of neurons
// bsize not used for now, only used if exploit data parallelism in FW/BW
template<typename KernelClass, typename weight_fmt, typename LLConcateFmt,typename NLConcateFmt, typename NLAGFmt,
         typename InPipe, typename InSigPipe, typename InWPipe, typename InBPipe, typename OutPipe, typename abufPipe, 
         int LL_nn, int NL_nn>
struct MyAutorun_MMFW {
    void operator()() const {
        // on-chip arrays holding w,b,itm results
        [[intel::singlepump,
        intel::fpga_memory("BLOCK_RAM"),
        intel::numbanks(NL_nn),
        intel::max_replicates(NL_nn)]]
        weight_fmt W[NL_nn]; //the LL_nn dimenssion is wrapped in weight_fmt struct.
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(NL_nn)]]
        float B[NL_nn];

        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        LLConcateFmt In[1];
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        NLConcateFmt Out[1];
        // PRINTF("in fpga kernel ...\n");

        while(1){
            // PRINTF("itm kernel, j: %d\n",j);
            In[0] = InPipe::read();
            for (size_t i=0; i<NL_nn; i++){
                W[i]=InWPipe.read();
                B[i]=InBPipe.read();
            }
            for (size_t i=0; i<LL_nn; i++){
                #pragma unroll
                for (size_t j=0; j<NL_nn; j++){
                    Out[0].s[j] += In[i].s * W[j].w[i];
                    Out0[0].snt[j] += In[i].snt * W[j].w[i];
                }
            }
            #pragma unroll
            for (size_t i=0; i<NL_nn; i++){
                // bias 
                Out[0].s[i] += B[i]; 
                Out[0].snt[i] += B[i]; 
                //activation (Relu)
                if (Out[0].s[i]<0) Out[0].s[i]=0;
                if (Out[0].snt[i]<0) Out[0].snt[i]=0;
            }         
            OutPipe::write(Out[0]); 
            NLAGFmt actdata;   
            for (size_t i=0; i<NL_nn; i++){
                actdata.s[i]=Out[0].s[i];
            }  
            abufPipe::write(actdata);
        }
    }
};

// The only diff with MyAutorun_MMFW is the lack of abufPipe since this is the output layer (OL) leading to
// the obj function, which is used to generate act gradients, not acts.
template<typename KernelClass, typename weight_fmt, typename LLConcateFmt,typename NLConcateFmt, 
         typename InPipe, typename InSigPipe, typename InWPipe, typename InBPipe, typename OutPipe, 
         int LL_nn, int NL_nn>
struct MyAutorun_MMFW_OL {
    void operator()() const {
        // on-chip arrays holding w,b,itm results
        [[intel::singlepump,
        intel::fpga_memory("BLOCK_RAM"),
        intel::numbanks(NL_nn),
        intel::max_replicates(NL_nn)]]
        weight_fmt W[NL_nn]; //the LL_nn dimenssion is wrapped in weight_fmt struct.
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(NL_nn)]]
        float B[NL_nn];

        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        LLConcateFmt In[1];
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        NLConcateFmt Out[1];
        // PRINTF("in fpga kernel ...\n");

        while(1){
            // PRINTF("itm kernel, j: %d\n",j);
            In[0] = InPipe::read();
            for (size_t i=0; i<NL_nn; i++){
                W[i]=InWPipe.read();
                B[i]=InBPipe.read();
            }
            for (size_t i=0; i<LL_nn; i++){
                #pragma unroll
                for (size_t j=0; j<NL_nn; j++){
                    Out[0].s[j] += In[i].s * W[j].w[i];
                    Out0[0].snt[j] += In[i].snt * W[j].w[i];
                }
            }
            #pragma unroll
            for (size_t i=0; i<NL_nn; i++){
                // bias 
                Out[0].s[i] += B[i]; 
                Out[0].snt[i] += B[i]; 
                //activation (Relu)
                if (Out[0].s[i]<0) Out[0].s[i]=0;
                if (Out[0].snt[i]<0) Out[0].snt[i]=0;
            }         
            OutPipe::write(Out[0]); 
        }
    }
};

// objective function. (r + gamma * argmaxQ_snt - Q_s)
// OLConcateFmt = L3ItmConcate, which is the format of data in InQsPipe
// the format of data in InRDonePipe is RDone
// the format of data in OutPipe is <OL>AG (L3AG when called)
// the format of data in DOLPipe is <OL>AG (L3AG when called)
template<typename KernelClass, typename OLConcateFmt, typename OLAG,
         typename InQsPipe, typename InRDonePipe, typename OutPipe, typename DOLPipe, //output-layer act derivative
         int OL_nn>
struct MyAutorun_OBJ {
    void operator()() const {
        // on-chip arrays
        float gamma = 0.3; //hyperparamter set at compile time
        [[intel::fpga_register]]
        OLAG out[1];
        while(1){
            // PRINTF("itm kernel, j: %d\n",j);
            auto rdone = InRDonePipe::read();
            OLConcateFmt Qs= InQsPipe::read(); 
            float maxQsnt=-10000;
            for (size_t i=0; i<OL_nn; i++){
                if (Qs.snt[i]>maxQsnt) maxQsnt=Qs.snt[i];
            }
            #pragma unroll
            for (size_t i=0; i<OL_nn; i++){
                out.s[i]= rdone.r + (1-rdone.done) * gamma * maxQsnt - Qs.s[i];
            }         
            OutPipe::write(out); 
        }
    }
};

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