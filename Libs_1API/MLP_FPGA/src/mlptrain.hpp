#ifndef __MLPTRAIN_HPP__
#define __MLPTRAIN_HPP__

// On devcloud:
// #include <sycl/sycl.hpp>
// On local install:
#include <CL/sycl.hpp>
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
#define L1 4 //input dimension, size(state)
#define L2 4 //hidden dimension
#define L3 2 //output dimension, num actions
// #define NL_chunksize 32 //output dimension

// typedefs
// using weight_fmt = float; //using an array-struct for fasterr streaming
// using act_fmt = float;

// Just for inf
typedef struct {
	float s[L1]; //current state
    float snt[L1]; //next state
} StateConcate;
typedef struct {
	float s[L2]; //L2 act/gr of current state
    float snt[L2]; //L2 act/gr next state
} L2ItmConcate;
typedef struct {
	float s[L3]; //L3 act/gr of current state
    float snt[L3]; //L3 act/gr next state
} L3ItmConcate;
typedef struct {
	float w[L1];
} W1Fmt; //assume weights are float
typedef struct {
	float w[L2];
} W2Fmt; //assume weights are float
//for both inf and training
typedef struct { //L3
	float w[L3];
} W2TranspFmt; //assume weights are float
typedef struct {
	float s[L1]; //current state
} L1AG;
typedef struct {
	float s[L2]; //L2 act/gr of current state
} L2AG;
typedef struct {
	float s[L3]; //L3 act/gr of current state
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
using ReadW1Pipe = ext::intel::pipe<class ReadW1PipeClass, W1Fmt,2*L2>; //<name, datatype, pipe depth>
using ReadB1Pipe = ext::intel::pipe<class ReadB1PipeClass, float,2*L2>;
//sigals weight sync, 1 means yes (stream in new weights to autorun kernels), 0 means no (keep weights static)
using L1FWSigPipe = ext::intel::pipe<class L1SigPipeClass, bool,4+L2>; 
using ReadW2Pipe = ext::intel::pipe<class ReadW2PipeClass, W2Fmt,2*L3>;
using ReadB2Pipe = ext::intel::pipe<class ReadB2PipeClass, float,2*L3>;
//sigals weight sync, 1 means yes (stream in new weights to autorun kernels), 0 means no (keep weights static)
using L2FWSigPipe = ext::intel::pipe<class L2SigPipeClass, bool,4+L3>; 
using A0Pipe = ext::intel::pipe<class A0PipeClass, L1AG,8>; //its depth should be >= processed sub-batch size (chunk size)
// from producer directly to obj, depth should be >= processed batch size (total count)
using RDonePipe = ext::intel::pipe<class RDonePipeClass, RDone,64>; 


using L12Pipe = ext::intel::pipe<class L12PipeClass, L2ItmConcate,1>;
using A1Pipe = ext::intel::pipe<class A1PipeClass, L2AG,8>; //its depth should be >= processed sub-batch size (chunk size)
using ActDrFWPipe= ext::intel::pipe<class ActDrFWPipeClass, L2AG,8>; 

using L23Pipe = ext::intel::pipe<class L23PipeClass, L3ItmConcate,1>;

using L32Pipe = ext::intel::pipe<class L32PipeClass, L3AG,1>;
using D2Pipe = ext::intel::pipe<class D2PipeClass, L3AG,8>; //its depth should be >= processed sub-batch size (chunk size)
using ReadW2bwPipe = ext::intel::pipe<class ReadW2bwPipeClass, W2TranspFmt,L2>;
using L2BWSigPipe = ext::intel::pipe<class L2BWSigPipeClass, bool,8+L2>; 

using D1Pipe = ext::intel::pipe<class D1PipeClass, L2AG,8>; //its depth should be >= processed sub-batch size (chunk size)

// using ConsumePipe = ext::intel::pipe<class ConsumePipe1Class, act_fmt,16>; 
// using writeW1Pipe = ext::intel::pipe<class writeW1PipeClass, W1Fmt,L2>; //<name, datatype, pipe depth>
using writeW1Pipe = ext::intel::pipe<class writeW1PipeClass, L2AG,L1>; //<name, datatype, pipe depth>
using writeB1Pipe = ext::intel::pipe<class writeB1PipeClass, float,L2>;
// using writeW2Pipe = ext::intel::pipe<class writeW2PipeClass, W2Fmt,L3>;
using writeW2Pipe = ext::intel::pipe<class writeW2PipeClass, L3AG,L2>;
using writeB2Pipe = ext::intel::pipe<class writeB2PipeClass, float,L3>;


// CPU->FPGA, producer for FW layers and LOSS (objctv) 
// assume input data format is already handled at host (inputs/weights/<r,done> are arrays of structs, biases are arrays of floats)


template<typename OutPipeS, typename OutPipeW1, typename OutPipeW2, typename OutPipeB1, typename OutPipeB2, typename OutSigPipeW1, typename OutSigPipeW2, typename A0bufPipe, typename rdPipe, typename OutPipeW2T, typename OutSigPipeW2T>
event Submit_Producer(queue &q, StateConcate *state_in_buf, W1Fmt *w1_buf, W2Fmt *w2_buf, float *bias1_buf, float *bias2_buf, RDone *rdone_buf, W2TranspFmt *w2t_buf,
                      size_t size, bool stream_w) { //size is (sub-)batch size here, not state size
    
    // ext::oneapi::experimental::printf("***submit the producer\n");
    return q.single_task<P>([=]() [[intel::kernel_args_restrict]] {
        host_ptr<StateConcate> state_in(state_in_buf);
        host_ptr<W1Fmt> w1_in(w1_buf);
        host_ptr<W2Fmt> w2_in(w2_buf);
        host_ptr<float> b1_in(bias1_buf);
        host_ptr<float> b2_in(bias2_buf);
        host_ptr<RDone> rd_in(rdone_buf);
        host_ptr<W2TranspFmt> w2t_in(w2t_buf);
        
        for (size_t i = 0; i < size; i++) {
            OutPipeS::write(state_in[i]); 
            
            L1AG a0;
            #pragma unroll
            for (size_t j=0; j<L1; j++){
                a0.s[j]=state_in[i].s[j];
                
            }
            // A0bufPipe::write(state_in[i].s);
            A0bufPipe::write(a0);
            rdPipe::write(rd_in[i]);
        }  
        // read weights only if signaled to do so  
        if (stream_w){
            for (size_t i = 0; i < L2; i++){
                // ext::oneapi::experimental::printf("***writing W1\n");
                OutPipeW1::write(w1_in[i]); 
                // for (size_t j=0; j<L1; j++){
                //     std::cout<<w1_in[i].w[j]<<' ';
                // }
                OutPipeB1::write(b1_in[i]);
                // std::cout<<"b_in[i]: "<<b1_in<<'\n';
                OutSigPipeW1::write(1);
            }
            // ext::oneapi::experimental::printf("***writing W2\n");
            for (size_t i = 0; i < L3; i++){
                OutPipeW2::write(w2_in[i]); 
                OutPipeB2::write(b2_in[i]);
                OutSigPipeW2::write(1);
            }
            // ext::oneapi::experimental::printf("***writing W3\n");
            for (size_t i = 0; i < L2; i++){
                OutPipeW2T::write(w2t_in[i]); 
                OutSigPipeW2T::write(1);
            }
            // ext::oneapi::experimental::printf("***wrote Ws\n");
        } 
        else{OutSigPipeW1::write(0);OutSigPipeW2::write(0);OutSigPipeW2T::write(0);} 
    
        // PRINTF("***done with the producer\n");
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

        while(1){
            // PRINTF("itm kernel, j: %d\n",j);
            PRINTF("\nFPGA: FW pipe reading weights and bias\n");
            // PRINTF("FPGA: FW pipe reading s and snts:\n");
            In[0] = InPipe::read();
            // for (size_t i=0; i<LL_nn; i++){
                // PRINTF("%f ",In[0].s[i]); //checked-correct
                // PRINTF("%f ",In[0].snt[i]); //checked-correct
            // }
            bool rwf = InSigPipe::read();
            if (rwf){
                for (size_t i=0; i<NL_nn; i++){
                    W[i]=InWPipe::read();
                    B[i]=InBPipe::read();
                }
            }
            // for (size_t i=0; i<L1; i++){
            //     for (size_t j=0; j<L2; j++){
            //         PRINTF("%f ",W[j].w[i]); //checked-correct
            //     }
            //     PRINTF("\n");
            // }
            // for (size_t j=0; j<L2; j++){PRINTF("%f ",B[j]);} //checked-correct
            PRINTF("FPGA: Computing FW MM\n");
            #pragma unroll
            for (size_t j=0; j<NL_nn; j++){
                // init, +bias 
                Out[0].s[j]=B[j];
                Out[0].snt[j]= B[j]; 
            }
            for (size_t i=0; i<LL_nn; i++){
                #pragma unroll
                for (size_t j=0; j<NL_nn; j++){
                    // if(j==0)Out[0].s[j]=0;
                    Out[0].s[j] += In[0].s[i] * W[j].w[i];
                    Out[0].snt[j] += In[0].snt[i] * W[j].w[i];
                }
            }
            PRINTF("FPGA: FW add bias and relu\n");
            #pragma unroll
            for (size_t i=0; i<NL_nn; i++){
                // bias 
                // Out[0].s[i] += B[i]; 
                // Out[0].snt[i] += B[i]; 
                //activation (Relu)
                if (Out[0].s[i]<0) Out[0].s[i]=0;
                if (Out[0].snt[i]<0) Out[0].snt[i]=0;
            }      
            // PRINTF("\nFPGA: L1 FW outputs from s and snt:\n");   
            // for (size_t i=0; i<NL_nn; i++){
            //     PRINTF("%f ", Out[0].s[i]);
            //     PRINTF("%f ", Out[0].snt[i]);
            // }  //checked-correct
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
            bool rwf = InSigPipe::read();
            if (rwf){
                for (size_t i=0; i<NL_nn; i++){
                    W[i]=InWPipe::read();
                    B[i]=InBPipe::read();
                }
            }
            #pragma unroll
            for (size_t j=0; j<NL_nn; j++){
                // init, +bias 
                Out[0].s[j]=B[j];
                Out[0].snt[j]= B[j]; 
            }
            for (size_t i=0; i<LL_nn; i++){
                #pragma unroll
                for (size_t j=0; j<NL_nn; j++){
                    Out[0].s[j] += In[0].s[i] * W[j].w[i];
                    Out[0].snt[j] += In[0].snt[i] * W[j].w[i];
                }
            }
            #pragma unroll
            for (size_t i=0; i<NL_nn; i++){
                // bias 
                // Out[0].s[i] += B[i]; 
                // Out[0].snt[i] += B[i]; 
                //activation (Relu)
                if (Out[0].s[i]<0) Out[0].s[i]=0;
                if (Out[0].snt[i]<0) Out[0].snt[i]=0;
            }     
            // PRINTF("\nFPGA: L2 FW outputs from s and snt:\n");   
            // for (size_t i=0; i<NL_nn; i++){
            //     PRINTF("%f ", Out[0].s[i]);
            //     PRINTF("%f ", Out[0].snt[i]);
            // }  //checked - correct
            OutPipe::write(Out[0]); 
        }
    }
};

// objective function. (r + gamma * argmaxQ_snt - Q_s)
// OLConcateFmt = L3ItmConcate (OLItmConcate), which is the format of data in InQsPipe
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
            PRINTF("\nFPGA: OBJ outputs:\n");  
            #pragma unroll
            for (size_t i=0; i<OL_nn; i++){
                out[0].s[i] = (rdone.r + (1-rdone.done) * gamma * maxQsnt - Qs.s[i]) * (Qs.s[i] > 0 ? 1 : 0);
                // (Qs.s[i] > 0 ? 1 : 0) is the ct derivative
                PRINTF("%f ", out[0].s[i]); //checked - correct
            }         
            OutPipe::write(out[0]); 
            DOLPipe::write(out[0]); 
        }
    }
};

// weight_fmt is W2TranspFmt, W size L2, each W2TranspFmt size L3

// LLConcateFmt is for InPipe, NLConcateFmt is for OutPipe
// InSIgPipe is used to signal InWPipe
// the last BW function, only outputs L2 delta (derivatives)
// 1*L3, L3*L2 - LL_nn=L3, NL_nn=L2
// LLConcateFmt is L3AG, NLConcateFmt is L2AG 
template<typename KernelClass, typename weight_fmt, typename LLConcateFmt,typename NLConcateFmt, 
         typename InPipe, typename InSigPipe, typename InWPipe, typename OutPipe, 
         int LL_nn, int NL_nn>
struct MyAutorun_MMBW_OL {
    void operator()() const {
        // on-chip arrays holding w,b,itm results
        [[intel::singlepump,
        intel::fpga_memory("BLOCK_RAM"),
        intel::numbanks(NL_nn),
        intel::max_replicates(NL_nn)]]
        weight_fmt W[NL_nn]; //the LL_nn dimenssion is wrapped in weight_fmt struct.

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
            bool rwf = InSigPipe::read();
            // PRINTF("\nFPGA: weights in transposed:\n");  //checked - correct
            if (rwf){ //read weights in transposed
                for (size_t i=0; i<NL_nn; i++) { //L2
                    W[i]=InWPipe::read();
                    // for(size_t j=0; j<LL_nn; j++){ //L3
                        // PRINTF("%f ",W[i].w[j]);  //checked - correct
                    // }
                    // PRINTF("\n"); 
                }
            }
            // PRINTF("\nFPGA: BW outputs:\n"); //checked - correct
            for (size_t j=0; j<LL_nn; j++)Out[0].s[j]=0;
            for (size_t i=0; i<LL_nn; i++){ //L3
                #pragma unroll
                for (size_t j=0; j<NL_nn; j++){ //L2
                    Out[0].s[j] += In[0].s[i] * W[j].w[i];
                }
            }
            // for (size_t i=0; i<NL_nn; i++){
            //     PRINTF("%f ", Out[0].s[i]);//checked - correct
            // }      
            
            OutPipe::write(Out[0]); 
        }
    }
};

// act: LL_nn, delta: NL_nn
// LL*1, 1*NL
// output (OutFmt) wraps NL as a struct array (width of OutPipe = NL)
// OutnextWAPipe is used to send hiddenlayer outputs (size: LL_nn) to the previous-layers' WA modules, as those need to multiply BW gradients with act-derivetives of thhes hidden outputs.
template<typename KernelClass, typename ActFmt, typename GrdFmt, typename OutFmt, 
         typename ActPipe, typename GrdPipe, typename OutWPipe, typename OutBPipe, typename OutnextWAPipe,
         int LL_nn, int NL_nn>
struct MyAutorun_MMWA_OL {
    void operator()() const {
        // on-chip arrays holding w,b,itm results
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        OutFmt OutW[LL_nn];
        // PRINTF("in fpga kernel ...\n");
        [[intel::fpga_register]]
        float OutB[NL_nn];

        while(1){
            ActFmt act_arr = ActPipe::read();
            GrdFmt grd_arr = GrdPipe::read();
            // PRINTF("\nFPGA: WA outputs for %d * %d:\n",LL_nn,NL_nn);  

            for (size_t i=0; i<LL_nn; i++){
                #pragma unroll
                for (size_t j=0; j<NL_nn; j++){ 
                    OutW[i].s[j] = act_arr.s[i] * grd_arr.s[j];
                    // PRINTF("%f ", OutW[i].s[j]); //checked - correct 
                }
                // PRINTF("\n");
                OutWPipe::write(OutW[i]); 
            }                
            for (size_t j=0; j<NL_nn; j++){ 
                OutB[j] = grd_arr.s[j];
                OutBPipe::write(OutB[j]); 
            }
            // for (size_t i=0; i<LL_nn; i++){
            //     OutWPipe::write(OutW[i]);     
            // }     
            OutnextWAPipe::write(act_arr);
        }
    }
};


// act: LL_nn, delta: NL_nn
// LL*1, 1*NL
// output (OutFmt) wraps NL as a struct array (width of OutPipe = NL)
// InLastWAPipe is used to rerceive hiddenlayer outputs (size: NL_nn) from the next-layers' WA modules
template<typename KernelClass, typename ActFmt, typename GrdFmt, typename OutFmt, 
         typename ActPipe, typename GrdPipe, typename InLastWAPipe, typename OutWPipe, typename OutBPipe,
         int LL_nn, int NL_nn>
struct MyAutorun_MMWA {
    void operator()() const {
        // on-chip arrays holding w,b,itm results
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        OutFmt OutW[LL_nn]; //L1
        // PRINTF("in fpga kernel ...\n");
        [[intel::fpga_register]]
        float OutB[NL_nn]; //L2

        while(1){
            ActFmt act_arr = ActPipe::read();
            GrdFmt grd_arr = GrdPipe::read();
            GrdFmt h_out = InLastWAPipe::read();
            PRINTF("\nFPGA: Bias outputs for %d * 1:\n",NL_nn);  
            // #pragma unroll
            for (size_t j=0; j<NL_nn; j++){ //L2
                PRINTF("%f ", grd_arr.s[j]); 
                grd_arr.s[j] *= (h_out.s[j] > 0 ? 1 : 0); //hiddenOutput: activation derivative of L1-FW output, used in WA1
                OutB[j] = grd_arr.s[j];
                // PRINTF("%f ", h_out.s[j]); //checked-correct
                // PRINTF("%f ", OutB[j]); //checked-correct
                OutBPipe::write(OutB[j]); 
            }
            // PRINTF("\nFPGA: WA outputs for %d * %d:\n",LL_nn,NL_nn);  
            for (size_t i=0; i<LL_nn; i++){//L1
                #pragma unroll
                for (size_t j=0; j<NL_nn; j++){ //L2
                    OutW[i].s[j] = act_arr.s[i] * grd_arr.s[j];
                    // PRINTF("%f ", OutW[i].s[j]); //checked-correct
                }
                // PRINTF("\n");
                OutWPipe::write(OutW[i]); 
            }                

            // for (size_t i=0; i<LL_nn; i++){
            //     OutWPipe::write(OutW[i]);     
            // }     
            // OutnextWAPipe::write(act_arr);
        }
    }
};

// todo: update args weights format as AG, consistent with compute kernels
template <typename InW1Pipe,typename InW2Pipe, typename InB1Pipe,typename InB2Pipe>
event Submit_Consumer(queue& q, L2AG *wg1_buf, L3AG *wg2_buf, float *biasg1_buf, float *biasg2_buf, size_t size) {
  return q.single_task<C>([=]() [[intel::kernel_args_restrict]] {
    host_ptr<L2AG> wg1(wg1_buf); //needs to be intialized to all 0
    host_ptr<L3AG> wg2(wg2_buf); //needs to be intialized to all 0
    host_ptr<float> bg1(biasg1_buf); //needs to be intialized to all 0
    host_ptr<float> bg2(biasg2_buf); //needs to be intialized to all 0
    for (size_t i = 0; i < size; i++) {
        L2AG res1;
        // #pragma unroll
        // for (size_t k=0; k<L2; k++)res1[k]=0;
        L3AG res2;
        // #pragma unroll
        // for (size_t k=0; k<L3; k++)res2[k]=0;
        for (size_t j=0; j<L1; j++){
            L2AG w1_wg = InW1Pipe::read();
            #pragma unroll
            for (size_t k=0; k<L2; k++){ 
                res1.s[k] = w1_wg.s[k]; 
            }  
            *(wg1 + j)  = res1;

        }
        for (size_t j=0; j<L2; j++){
            L3AG w2_wg = InW2Pipe::read();
            #pragma unroll
            for (size_t k=0; k<L3; k++){
                res2.s[k] = w2_wg.s[k];
            }    
            *(wg2 + j)  = res2;
        }
        for (size_t j=0; j<L2; j++){
            float b1_wg = InB1Pipe::read();
            *(bg1 + j) = b1_wg;  
        }
        for (size_t j=0; j<L3; j++){
            float b2_wg = InB2Pipe::read();
            *(bg2 + j) = b2_wg;  
        }
    }
  });
}


#endif /* __MLPTRAIN_HPP__ */