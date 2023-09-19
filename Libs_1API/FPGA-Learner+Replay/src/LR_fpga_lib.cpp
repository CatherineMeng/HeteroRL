#include <CL/sycl.hpp>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iomanip>  // for std::setprecision
#include <iostream>
#include <vector>

#include "include/exception_handler.hpp"
#include "autorun.hpp"

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;


#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }

// Sum Tree metadata
#define K 8//fanout 
#define D 4 //depth with root
#define Lev1_Width 4
#define Lev2_Width 16
#define Lev3_Width 64

// DNN policy metadata
//2-layer MLP, input (L1), hidden (L2), output (L3)
#define L1 4 //input dimension, size(state)
#define L2 4 //hidden dimension
#define L3 2 //output dimension, num actions

// hardware parallelism
#define F2_FW 4 //FW: hidden dimension unroll factor
#define F3_FW 1 //FW: output dimension unroll factor
#define F2_BW 4 //BW: hidden dimension unroll factor
#define F2_WA 2 //WA-layer 1: hidden dimension unroll factor
#define F3_WA 2 //WA-layer 2: output dimension unroll factor

using fixed_root = float;
using fixed_l1 = float;
using fixed_l2 = float;
using fixed_l3 = float;

using fixed_upd = float;
using fixed_insrt = float;

using fixed_bool = bool;

// data pack (instruction) transfered along sibling iterators. They are used in fpga hardware compilation.
// update_offsets are calculated on the host based on the difference between sampled value and value to be updated
// insertion is performed by get_priority followed by update from host (1. get_priority to get the old TD value, 2. update new TD value)
typedef struct {
	// ---sampling---
	fixed_bool sampling_flag;
	int start; //the start index of next layer sibling iteration
	fixed_root newx; //the new prefix sum value to be found in the next layer
	// ---update---
	fixed_bool update_flag;
	// int update_index_array[D]; //root lev index =0, so [0, ...]
	// fixed_root update_offset_array[D]; //root lev index =0
    std::array<int, D> update_index_array;//root lev index =0, so [0, ...]
    std::array<fixed_root, D> update_offset_array;//root lev index =0
    // ---tree initialization (do once)---
    fixed_bool init_flag;
} sibit_io;

// A data point stoerd in replay memory data storage
typedef struct {
	// ---sampling---
	std::array<float, L1> state;
    int action;
    std::array<float, L1> next_state;
    float reward;
    int done;
    float pr; 
    //no retrieval from tree, directly retrieve from data storage 
} experience;

// Intermediate results passed in DNN training
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
// just for obj
typedef struct {
	float r; //reward
    int done; //indicate whether terminal state
} RDone;

// class declarations for autorun kernels - PER
class Itm1;
class Itm2;
class Itm3;
class PR;
class CR;
// class declarations for autorun kernels - learner
class PL; //producer - learner
class MM_FW1;
class MM_FW2;
class OBJ;
class MM_BW;
class WA1;
class WA2;
class CL; //consumer - learner

// // the pipes used in PER
using ProducePipe = ext::intel::pipe<class ProducePipeClass, sibit_io,16>;
using ConsumePipe1 = ext::intel::pipe<class ConsumePipe1Class, int,16>; //sampled ind
// using ConsumePipe2 = ext::intel::pipe<class ConsumePipe2Class, fixed_l3,16>; //samapled val
// using ConsumePipe3 = ext::intel::pipe<class ConsumePipe3Class, fixed_l3,16>; //get_priority val for insertion
using L1_L2_Pipe = ext::intel::pipe<class L1_L2_PipeClass, sibit_io,16>;
using L2_L3_Pipe = ext::intel::pipe<class L2_L3_PipeClass, sibit_io,16>;

// // the pipes used in LNR
using SinPipe = ext::intel::pipe<class SinPipeClass, StateConcate,8>; 
using ReadW1Pipe = ext::intel::pipe<class ReadW1PipeClass, W1Fmt,2*L2>; 
using ReadB1Pipe = ext::intel::pipe<class ReadB1PipeClass, float,2*L2>;
using L1FWSigPipe = ext::intel::pipe<class L1SigPipeClass, bool,4+L2>; 
using ReadW2Pipe = ext::intel::pipe<class ReadW2PipeClass, W2Fmt,2*L3>;
using ReadB2Pipe = ext::intel::pipe<class ReadB2PipeClass, float,2*L3>;
using L2FWSigPipe = ext::intel::pipe<class L2SigPipeClass, bool,4+L3>; 
using A0Pipe = ext::intel::pipe<class A0PipeClass, L1AG,8>; 
using RDonePipe = ext::intel::pipe<class RDonePipeClass, RDone,64>; 
using L12Pipe = ext::intel::pipe<class L12PipeClass, L2ItmConcate,1>;
using A1Pipe = ext::intel::pipe<class A1PipeClass, L2AG,8>; 
using ActDrFWPipe= ext::intel::pipe<class ActDrFWPipeClass, L2AG,8>; 
using L23Pipe = ext::intel::pipe<class L23PipeClass, L3ItmConcate,1>;
using L32Pipe = ext::intel::pipe<class L32PipeClass, L3AG,1>;
using NupdPipe = ext::intel::pipe<class NupdPipeClass, L3AG,1>;
using D2Pipe = ext::intel::pipe<class D2PipeClass, L3AG,8>; 
using ReadW2bwPipe = ext::intel::pipe<class ReadW2bwPipeClass, W2TranspFmt,L2>;
using L2BWSigPipe = ext::intel::pipe<class L2BWSigPipeClass, bool,8+L2>; 
using D1Pipe = ext::intel::pipe<class D1PipeClass, L2AG,8>; 
using writeW1Pipe = ext::intel::pipe<class writeW1PipeClass, L2AG,L1>; 
using writeB1Pipe = ext::intel::pipe<class writeB1PipeClass, float,L2>;
using writeW2Pipe = ext::intel::pipe<class writeW2PipeClass, L3AG,L2>;
using writeB2Pipe = ext::intel::pipe<class writeB2PipeClass, float,L3>;



// Learner autorun structs

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
         int LL_nn, int NL_nn, int NL_F>
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
            PRINTF("\nFPGA: FW pipe reading weights and bias\n");
            In[0] = InPipe::read();
            bool rwf = InSigPipe::read();
            if (rwf){
                for (size_t i=0; i<NL_nn; i++){
                    W[i]=InWPipe::read();
                    B[i]=InBPipe::read();
                }
            }
            PRINTF("FPGA: Computing FW MM\n");
            #pragma unroll
            for (size_t j=0; j<NL_nn; j++){
                // init, +bias 
                Out[0].s[j]=B[j];
                Out[0].snt[j]= B[j]; 
            }
            for (size_t i=0; i<LL_nn; i++){
                for (size_t ii=0; ii<NL_nn/NL_F; ii++){
                    #pragma unroll
                    for (size_t j=0; j<NL_F; j++){
                        // if(j==0)Out[0].s[j]=0;
                        Out[0].s[ii*NL_F+j] += In[0].s[i] * W[ii*NL_F+j].w[i];
                        Out[0].snt[ii*NL_F+j] += In[0].snt[i] * W[ii*NL_F+j].w[i];
                    }
                }
            }
            PRINTF("FPGA: FW add bias and relu\n");
            #pragma unroll
            for (size_t i=0; i<NL_nn; i++){
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
         int LL_nn, int NL_nn, int NL_F>
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
                for (size_t ii=0; ii<NL_nn/NL_F; ii++){
                    #pragma unroll
                    for (size_t j=0; j<NL_F; j++){
                        Out[0].s[ii*NL_F+j] += In[0].s[i] * W[ii*NL_F+j].w[i];
                        Out[0].snt[ii*NL_F+j] += In[0].snt[i] * W[ii*NL_F+j].w[i];
                    }
                }
            }
            #pragma unroll
            for (size_t i=0; i<NL_nn; i++){
                if (Out[0].s[i]<0) Out[0].s[i]=0;
                if (Out[0].snt[i]<0) Out[0].snt[i]=0;
            }     
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
         typename NewPRPipe, //output for generating next update requests
         int OL_nn>
struct MyAutorun_OBJ {
    void operator()() const {
        // on-chip arrays
        float gamma = 0.3; //hyperparamter set at compile time
        [[intel::fpga_register]]
        OLAG out[1];
        while(1){
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
                PRINTF("%f ", out[0].s[i]); 
            }         
            OutPipe::write(out[0]); 
            DOLPipe::write(out[0]); 
            NewPRPipe::write(out[0]); 
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
         int LL_nn, int NL_nn, int NL_F>
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
        while(1){
            // PRINTF("itm kernel, j: %d\n",j);
            In[0] = InPipe::read();
            bool rwf = InSigPipe::read();
            // PRINTF("\nFPGA: weights in transposed:\n");  //checked - correct
            if (rwf){ //read weights in transposed
                for (size_t i=0; i<NL_nn; i++) { //L2
                    W[i]=InWPipe::read();
                }
            }
            // PRINTF("\nFPGA: BW outputs:\n"); //checked - correct
            for (size_t j=0; j<LL_nn; j++)Out[0].s[j]=0;
            for (size_t i=0; i<LL_nn; i++){ //L3
                for (size_t ii=0; ii<NL_nn/NL_F; ii++){
                    #pragma unroll
                    for (size_t j=0; j<NL_F; j++){ //L2
                        Out[0].s[ii*NL_F+j] += In[0].s[i] * W[ii*NL_F+j].w[i];
                    }
                }
            }
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
         int LL_nn, int NL_nn, int NL_F>
struct MyAutorun_MMWA_OL {
    void operator()() const {
        // on-chip arrays holding w,b,itm results
        [[intel::singlepump,
        intel::fpga_memory("MLAB"),
        intel::numbanks(1)]]
        OutFmt OutW[LL_nn];
        [[intel::fpga_register]]
        float OutB[NL_nn];

        while(1){
            ActFmt act_arr = ActPipe::read();
            GrdFmt grd_arr = GrdPipe::read();
            // PRINTF("\nFPGA: WA outputs for %d * %d:\n",LL_nn,NL_nn);  

            for (size_t i=0; i<LL_nn; i++){
                for (size_t ii=0; ii<NL_nn/NL_F; ii++){
                    #pragma unroll
                    for (size_t j=0; j<NL_F; j++){ 
                        OutW[i].s[ii*NL_F+j] = act_arr.s[i] * grd_arr.s[ii*NL_F+j];
                    }
                }
                OutWPipe::write(OutW[i]); 
            }                
            for (size_t j=0; j<NL_nn; j++){ 
                OutB[j] = grd_arr.s[j];
                OutBPipe::write(OutB[j]); 
            }    
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
         int LL_nn, int NL_nn, int NL_F>
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
                OutBPipe::write(OutB[j]); 
            }
            // PRINTF("\nFPGA: WA outputs for %d * %d:\n",LL_nn,NL_nn);  
            for (size_t i=0; i<LL_nn; i++){//L1
                for (size_t ii=0; ii<NL_nn/NL_F; ii++){//L2
                    #pragma unroll
                    for (size_t j=0; j<NL_F; j++){ //L2
                        OutW[i].s[ii*NL_F+j] = act_arr.s[i] * grd_arr.s[ii*NL_F+j];
                    }
                    OutWPipe::write(OutW[i]); 
                }

            }                
        }
    }
};


// data packs output by the FPGA kernels. They are only used in DoWorkMultiKernel_LR functions to realize
// pass-by-reference functionalities when using pybind, and do not affect fpga hardware compilation.
struct MultiKernel_out_LR {
    MultiKernel_out_LR(int bsize) {
        new_upd_reqs.resize(bsize);
        wg1.resize(L1);
        wg2.resize(L2);
        biasg1.resize(L2);
        biasg2.resize(L3);
    }
    // std::vector<int> sampled_idx;
    fixed_root root_pr; // for usrs to check total pr
    std::vector<L2AG>wg1;
    std::vector<L3AG>wg2;
    std::vector<float>biasg1;
    std::vector<float>biasg2;
    std::vector<sibit_io>new_upd_reqs;
};



class PER_LNR {
public:
    // constructor
    // ds_side: data storage size; bsize: sampling/update/training batch size
    PER_LNR(int ds_size, int bsize){

        // Create selector 
        #if FPGA_EMULATOR
            sycl::ext::intel::fpga_emulator_selector selector;
        #elif FPGA
            sycl::ext::intel::fpga_selector selector;
        #else
            sycl::default_selector selector;
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

        // create queue (private)
        q = sycl::queue(selector, exception_handler);
        // return q;s
        std::cout << "Constructor: Task queue created" << std::endl;

        // allocate the FPGA device DDR memory
        // T* out_ptr = malloc_device<T>(size, q);
        
        data_storage = malloc_device<experience>(ds_size, q);
        prs_buf = malloc_device<L3AG>(bsize, q);

        if(data_storage == nullptr) {
            std::cerr << "ERROR: failed to allocate space for 'data_storage'\n";
        }


        // Put data storage onto FPGA device
        // copy_host_to_device_event = q.memcpy(data_storage, in.data(), size*sizeof(float));

        // start the FPGA autorun kernels - PER (prioritized exp replay)
        MyAutorun_itm<Itm1, fixed_l1, fixed_root, ProducePipe, L1_L2_Pipe, 1, Lev1_Width>();
        MyAutorun_itm<Itm2, fixed_l2, fixed_l1, L1_L2_Pipe, L2_L3_Pipe, 2, Lev2_Width>();
        MyAutorun_lastlev<Itm3,fixed_l3, fixed_l2, L2_L3_Pipe, ConsumePipe1, Lev3_Width>();
        std::cout << "Constructor: Autorun started" << std::endl;

        // start the FPGA autorun kernels - LNR (learner)
        fpga_tools::Autorun<MM_FW1> ar_kernel1{selector, MyAutorun_MMFW<MM_FW1, W1Fmt, StateConcate,L2ItmConcate, L2AG,
                SinPipe, L1FWSigPipe, ReadW1Pipe, ReadB1Pipe, L12Pipe, A1Pipe, L1, L2, F2_FW>{}};

        fpga_tools::Autorun<MM_FW2> ar_kernel2{selector, MyAutorun_MMFW_OL<MM_FW2, W2Fmt, L2ItmConcate, L3ItmConcate,
                L12Pipe, L2FWSigPipe, ReadW2Pipe, ReadB2Pipe, L23Pipe, L2, L3, F3_FW>{}};

        fpga_tools::Autorun<OBJ> ar_kernel3{selector, MyAutorun_OBJ<OBJ, L3ItmConcate, L3AG, 
                L23Pipe, RDonePipe, L32Pipe, D2Pipe, NupdPipe, L3>{}};

        fpga_tools::Autorun<MM_BW> ar_kernel4{selector, MyAutorun_MMBW_OL <MM_BW, W2TranspFmt, L3AG, L2AG,
                L32Pipe, L2BWSigPipe, ReadW2bwPipe, D1Pipe, L3, L2, F2_BW>{}};

        fpga_tools::Autorun<WA2> ar_kernel6{selector, MyAutorun_MMWA_OL <WA2, L2AG, L3AG, L3AG,
                A1Pipe, D2Pipe, writeW2Pipe, writeB2Pipe, ActDrFWPipe, L2, L3, F3_WA>{}};//WA2

        fpga_tools::Autorun<WA1> ar_kernel5{selector, MyAutorun_MMWA <WA1, L1AG, L2AG, L2AG,
                A0Pipe, D1Pipe, ActDrFWPipe, writeW1Pipe, writeB1Pipe, L1, L2, F2_WA>{}}; //WA1 

    }

    //destructor
    ~PER_LNR(){
        // free(indices,q);
        free(data_storage,q);
        free(prs_buf,q);
    }

    template<typename KernelClass, typename TLevDType, typename ParTLevDType, typename InPipe, 
            typename OutPipe, int treelayer, int layersize>
    // void MyAutorun_itm (queue& q) {
    void MyAutorun_itm () {
        // Declare the SRAM array for the current tree layer
        q.submit([&](handler &h) {
            h.single_task<KernelClass>([=]() {
                [[intel::singlepump, intel::fpga_memory("MLAB"), intel::numbanks(1)]]
                TLevDType TLev[layersize];
                // initialize on-chip memory
                for (size_t j=0;j<layersize;j++){
                    TLev[j]=0;
                }
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
            });
        });
    }

    template<typename KernelClass, typename LastLevDType, typename ParTLevDType, 
            typename InPipe, typename OutPipe1, int layersize>
    void MyAutorun_lastlev (){
        q.submit([&](handler &h) {
            h.single_task<KernelClass>([=]() {
                [[intel::singlepump, intel::fpga_memory("MLAB"),intel::numbanks(1)]]
                LastLevDType TLev[layersize];
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
                    if (data_in.update_flag==1){
                        // ======== Update in a Sibling iterator ========
                        int idx = data_in.update_index_array[D-1];
                        LastLevDType val = data_in.update_offset_array[D-1];
                        TLev[idx]+=val;
                    }
                }
            });
        });
    }

    // CPU->FPGA
    // assume input data format is already handled at host (in_ptr is an array of sibit_io objects)
    // This is always for level 1, TLevDType = fixed_l1, layersize is the tree fanouts.
    template<typename OutPipe>
    event Submit_Producer_R(std::vector<sibit_io> &in_buf, size_t size) {
        buffer in_buf_knl(in_buf);
        return q.submit([&](handler &h) [[intel::kernel_args_restrict]] {
            accessor in(in_buf_knl, h, read_only);
            h.single_task<PR>([=] {
                for (size_t j = 0; j < size; j++) {
                    OutPipe::write(in[j]); 
                } 
            });      
        });
    }

    template <typename InPipe1, typename LastLevDType>
    event Submit_Consumer_R(size_t size, int *indices) {
        // buffer out_buf1_knl(out_buf1);
        
        return q.submit([&](handler &h) [[intel::kernel_args_restrict]] {
            // accessor out1(out_buf1_knl, h, write_only);
            h.single_task<CR>([=] {
                device_ptr<int> indices_d(indices);
                for (int i = 0; i < size; i++) {
                    auto data1 = InPipe1::read();
                    indices_d[i] = data1;
                }
            });
        });
    }

    void Init_Tree(){
    
        size_t batchsize = 2;       

        std::cout << "=== Running the kernel for initializing tree ===\n";

        std::vector<sibit_io> in;
        std::vector<int> sampled_idx;  
        int *indices = malloc_device<int>(batchsize, q);

        in.resize(2);
        sampled_idx.resize(2);
        // ======== Test ========
        in[0].sampling_flag=0;
        in[0].update_flag=0;
        in[0].init_flag=1;
        in[1].sampling_flag=0;
        in[1].update_flag=0;
        in[1].init_flag=1; 

        root_pr=0;
        event p_e = Submit_Producer_R<ProducePipe>(in, batchsize);   
        event c_e = Submit_Consumer_R<ConsumePipe1, fixed_l3>(
                                                    batchsize, indices);        
        p_e.wait();    // producer
        c_e.wait();   // consumer

        // events.wait();
        free(indices);
        printf("Init tree done. \n\n");
        
    }

// TODO: change input weights and biases to vectors, use explicit memcpy:
// double SubmitExplicitKernel(queue& q, std::vector<float>& in, size_t size) {
//   float * data_storage = malloc_device<float>(4*size, q);
// auto copy_host_to_device_event = q.memcpy(data_storage, in.data(), size*sizeof(float));
    template<typename OutPipeS, typename OutPipeW1, typename OutPipeW2, typename OutPipeB1, typename OutPipeB2, typename OutSigPipeW1, typename OutSigPipeW2, typename A0bufPipe, typename rdPipe, typename OutPipeW2T, typename OutSigPipeW2T>
    event Submit_Producer_L(StateConcate *state_in_buf, W1Fmt*w1_buf, W2Fmt*w2_buf, float*bias1_buf, float*bias2_buf, RDone *rdone_buf, W2TranspFmt*w2t_buf,
                        size_t size, bool stream_w) {
        return q.single_task<PL>([=]() [[intel::kernel_args_restrict]] {
            device_ptr<StateConcate> state_in(state_in_buf);
            device_ptr<W1Fmt> w1_in(w1_buf);
            device_ptr<W2Fmt> w2_in(w2_buf);
            device_ptr<float> b1_in(bias1_buf);
            device_ptr<float> b2_in(bias2_buf);
            device_ptr<RDone> rd_in(rdone_buf);
            device_ptr<W2TranspFmt> w2t_in(w2t_buf);
            
            for (size_t i = 0; i < size; i++) {
                OutPipeS::write(state_in[i]); 
                L1AG a0;
                #pragma unroll
                for (size_t j=0; j<L1; j++){
                    a0.s[j]=state_in[i].s[j];
                    
                }
                A0bufPipe::write(a0);
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
                for (size_t i = 0; i < L2; i++){
                    OutPipeW2T::write(w2t_in[i]); 
                    OutSigPipeW2T::write(1);
                }
            } 
            else{OutSigPipeW1::write(0);OutSigPipeW2::write(0);OutSigPipeW2T::write(0);} 
        });
    }

    template <typename InW1Pipe,typename InW2Pipe, typename InB1Pipe,typename InB2Pipe, typename NupdPipe>
    // event Submit_Consumer_L(std::vector<L2AG>&wg1_buf, std::vector<L3AG>&wg2_buf, std::vector<float>&biasg1_buf, std::vector<float>&biasg2_buf, L3AG *prss, size_t size) {
    event Submit_Consumer_L(L2AG *wg1_buf, L3AG *wg2_buf, float *biasg1_buf, float *biasg2_buf, L3AG *prss, size_t size) {
    return q.single_task<CL>([=]() [[intel::kernel_args_restrict]] {
        host_ptr<L2AG> wg1(wg1_buf); 
        host_ptr<L3AG> wg2(wg2_buf); 
        host_ptr<float> bg1(biasg1_buf); 
        host_ptr<float> bg2(biasg2_buf); 
        for (size_t i = 0; i < size; i++) {
            L2AG res1;
            L3AG res2;
            L3AG new_upd_pr = NupdPipe::read();
            *(prss + i)  = new_upd_pr;
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


    // Tin:sibit_io, Tout: fixed_l3
    // An iteration: input update requests & input insertion requests -> update, barrier, sample, barrier, insert, learner -> output update requests & output new DNN weights
    MultiKernel_out_LR DoWorkMultiKernel_LR(std::vector<sibit_io> &in_sample_update, std::vector<sibit_io> &in_insert, std::vector<experience> &in_data, 
                            std::vector<W1Fmt>w1_bufv, std::vector<W2Fmt>w2_bufv, std::vector<float>bias1_bufv, std::vector<float>bias2_bufv, std::vector<W2TranspFmt>w2t_bufv, 
                            // std::vector<L2AG>wg1_buf, std::vector<L3AG>wg2_buf, std::vector<float>biasg1_buf, std::vector<float>biasg2_buf, //to be incldued in output struct
                            int batchsize_sample_update, int batchsize_insert) {
        // timing data
        // std::vector<double> latency_ms(iterations);
        // std::vector<double> process_time_ms(iterations);
        // auto start = high_resolution_clock::now();

        MultiKernel_out_LR outp(batchsize_sample_update);
        int *indices = malloc_device<int>(batchsize_sample_update, q);
        W1Fmt *w1_buf = malloc_device<W1Fmt>(L2, q);
        W2Fmt *w2_buf = malloc_device<W2Fmt>(L3, q);
        float *bias1_buf = malloc_device<float>(L2, q); 
        float *bias2_buf = malloc_device<float>(L3, q);
        W2TranspFmt *w2t_buf = malloc_host<W2TranspFmt>(L2, q);
        auto weightcopy_host_to_device_event1 = q.memcpy(w1_buf, w1_bufv.data(), L2*sizeof(W1Fmt));
        auto weightcopy_host_to_device_event2 = q.memcpy(w2_buf, w2_bufv.data(), L3*sizeof(W2Fmt));
        auto weightcopy_host_to_device_event3 = q.memcpy(w2t_buf, w2t_bufv.data(), L2*sizeof(W2TranspFmt));
        auto weightcopy_host_to_device_event4 = q.memcpy(bias1_buf, bias1_bufv.data(), L2*sizeof(float));
        auto weightcopy_host_to_device_event5 = q.memcpy(bias2_buf, bias2_bufv.data(), L3*sizeof(float));
        StateConcate *state_in_buf = malloc_device<StateConcate>(batchsize_sample_update, q);
        RDone *rdone_buf = malloc_device<RDone>(batchsize_sample_update, q);
    
        L2AG *wg1_buf;
        L3AG *wg2_buf;
        float *biasg1_buf;
        float *biasg2_buf;
        if ((wg1_buf = malloc_host<L2AG>(L1, q)) == nullptr || (wg2_buf = malloc_host<L3AG>(L2, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for wg1/wg2_buf\n";
        }
        if ((biasg1_buf = malloc_host<float>(L2, q)) == nullptr || (biasg2_buf= malloc_host<float>(L3, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for bg1/bg2_buf\n";
        }

        
        // ------- start update -------
        for (size_t i = 0; i < batchsize_sample_update; i++){
            assert(in_sample_update[i].update_flag==1);
            // std::cout<<"root_ptr = "<<root_pr<<" ";
            root_pr+=in_sample_update[i].update_offset_array[0];
            data_storage[in_sample_update[i].update_index_array[D-1]].pr += in_sample_update[i].update_offset_array[0]; //should this be in q.submit?
        }
        event p_eu = Submit_Producer_R<ProducePipe>(in_sample_update, batchsize_sample_update);   
        event c_eu = Submit_Consumer_R<ConsumePipe1,  fixed_l3>(
                                                    batchsize_sample_update, indices);
        p_eu.wait();    
        c_eu.wait();  
        // ------- barrier end update -------

        // ------- start sample -------
        for (size_t ii=0; ii<batchsize_sample_update; ii++){
            in_sample_update[ii].sampling_flag=1;
            in_sample_update[ii].update_flag=0;
            in_sample_update[ii].init_flag=0;
            in_sample_update[ii].start=0;
            // r is a random float between 0 and 1
            // float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            in_sample_update[ii].newx=root_pr * (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
        }
        event p_es = Submit_Producer_R<ProducePipe>(in_sample_update, batchsize_sample_update);   
        event c_es = Submit_Consumer_R<ConsumePipe1, fixed_l3>(
                                                    batchsize_sample_update, indices);
        p_es.wait();    
        c_es.wait();   
        // ------- barrier end sample -------

        // ------- start insert, use sampled idx for sampling, and learn -------
        // sampling from data storage
        std::vector<int> actions(batchsize_sample_update);
        std::vector<float> old_prs(batchsize_sample_update);
        for (size_t i=0; i<batchsize_sample_update; i++){ //should these be in q.submit?
            experience exp = data_storage[indices[i]];
            for (size_t j=0; j<L1; j++) {
                state_in_buf[i].s[j]=exp.state[j];
                state_in_buf[i].snt[j]=exp.next_state[j];
            }
            rdone_buf[i].r = exp.reward;
            rdone_buf[i].done = exp.done;
            actions[i]=exp.action;
            old_prs[i]=exp.pr;
        }
        // insertion
        for (size_t i=0; i<batchsize_insert; i++){ //should these be in q.submit?
            data_storage[i]=in_data[i];
        }
        event p_ei = Submit_Producer_R<ProducePipe>(in_insert, batchsize_insert);   
        event c_ei = Submit_Consumer_R<ConsumePipe1, fixed_l3>(
                                                    batchsize_insert, indices);
        // use sampled data for training
        weightcopy_host_to_device_event1.wait();
        weightcopy_host_to_device_event2.wait();
        weightcopy_host_to_device_event3.wait();
        weightcopy_host_to_device_event4.wait();
        weightcopy_host_to_device_event5.wait();
        event p_el = Submit_Producer_L<SinPipe,ReadW1Pipe,ReadW2Pipe,
                                    ReadB1Pipe,ReadB2Pipe,L1FWSigPipe,L2FWSigPipe,A0Pipe,RDonePipe,ReadW2bwPipe,L2BWSigPipe>
                                    (state_in_buf , w1_buf , w2_buf , bias1_buf , bias2_buf , rdone_buf , w2t_buf , batchsize_sample_update, 1); //rest of chunks is 0   
        event c_el = Submit_Consumer_L<writeW1Pipe,writeW2Pipe,writeB1Pipe,writeB2Pipe,NupdPipe>(wg1_buf,wg2_buf,biasg1_buf,biasg2_buf,  prs_buf, batchsize_sample_update);
        
        // generate new update requests
        for (size_t i=0; i<batchsize_sample_update; i++){
            outp.new_upd_reqs[i].init_flag=0;
            outp.new_upd_reqs[i].sampling_flag=0;
            outp.new_upd_reqs[i].update_flag=1;
            outp.new_upd_reqs[i].update_index_array[0]=0;
            outp.new_upd_reqs[i].update_index_array[1]=(indices[i]/K)/K;
            outp.new_upd_reqs[i].update_index_array[2]=indices[i]/K;
            outp.new_upd_reqs[i].update_index_array[3]=indices[i];
            // new pr update offsets: new pr - old pr
            // for (size_t ii=0; ii<D; ii++) outp.new_upd_reqs[i].update_offset_array[ii] = prs_buf[i].s[actions[i]] - old_prs[i];
            for (size_t ii=0; ii<D; ii++) outp.new_upd_reqs[i].update_offset_array[ii] = prs_buf[i].s[0] - old_prs[i]; //use 0 for now for data storage test
        }
        
        //generate new weight params (aggregated gradients)
        for (size_t i=0; i<L1; i++){
            for (size_t j=0; j<L2; j++){
                outp.wg1[i].s[j]=wg1_buf[i].s[j];   
            }
        }
        for (size_t i=0; i<L2; i++){
            for (size_t j=0; j<L3; j++){
                outp.wg2[i].s[j]=wg2_buf[i].s[j];   
            }
        }
        for (size_t j=0; j<L2; j++)outp.biasg1[j]=biasg1_buf[j];
        for (size_t j=0; j<L3; j++)outp.biasg2[j]=biasg2_buf[j];
        
        // wait on the producer/consumer kernel pair to finish 
        // p_el.wait();    // producer - swemu term.?
        // c_el.wait();   // consumer - swemu term.?
        // p_ei.wait();    
        // c_ei.wait(); 

        std::cout<<"Here\n";
        
        
        // outp.sampled_idx = sampled_idx;
        outp.root_pr = root_pr;

        return outp;
    // compute and print timing information
    // PrintPerformanceInfo<T>("Multi-kernel",total_count, latency_ms, process_time_ms);
    }

private:
    sycl::queue q;
    // int *indices;
    experience * data_storage;
    L3AG *prs_buf;

    fixed_root root_pr;
    
};