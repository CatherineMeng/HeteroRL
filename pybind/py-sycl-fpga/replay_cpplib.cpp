#include <CL/sycl.hpp>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iomanip>  // for std::setprecision
#include <iostream>
#include <vector>

#include "exception_handler.hpp"

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

// #define K 4 //fanout 
#define K 8//fanout 
#define D 4 //depth with root
// #define Lev1_Width 4
// #define Lev2_Width 16
// #define Lev3_Width 64
#define Lev1_Width 8
#define Lev2_Width 64
#define Lev3_Width 256

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
	// ---get_priority---
	fixed_bool get_priority_flag;
	int pr_idx; //the index used at the last tree level for obtaining old TD
    // ---tree initialization (do once)---
    fixed_bool init_flag;
    // member functionss used by pybind to access array variables. 
    // int get_upd_input_index(int idx){
    //     return update_index_array[idx];
    // }
    // fixed_root get_upd_offset_index(int idx){
    //     return update_offset_array[idx];
    // }
    // void set_upd_input_index(int idx, int val){
    //     update_index_array[idx]=idx;
    // }
    // void set_upd_offset_index(int idx, float val){
    //     update_offset_array[idx]=val;
    // }
} sibit_io;

// data packs output by the FPGA kernels. They are only used in DoWorkMultiKernel functions to realize
// pass-by-reference functionalities when using pybind, and do not affect fpga hardware compilation.
struct MultiKernel_out {
    MultiKernel_out(int bsize) {
        sampled_idx.resize(bsize);
        out_pr_sampled.resize(bsize);
        out_pr_insertion.resize(bsize);
    }
    std::vector<int> sampled_idx;
    std::vector<fixed_l3> out_pr_sampled;
    std::vector<fixed_l3> out_pr_insertion;
    fixed_root root_pr;
};


// class declarations for autorun kernels
class Itm1;
class Itm2;
class Itm3;

// class declarations for request submitters
class P;
class C;

// // the pipes used to produce/consume data
// try moving inside constructor
using ProducePipe = ext::intel::pipe<class ProducePipeClass, sibit_io,16>;
using ConsumePipe1 = ext::intel::pipe<class ConsumePipe1Class, int,16>; //sampled ind
using ConsumePipe2 = ext::intel::pipe<class ConsumePipe2Class, fixed_l3,16>; //samapled val
using ConsumePipe3 = ext::intel::pipe<class ConsumePipe3Class, fixed_l3,16>; //get_priority val for insertion
// internal pipes between kernels
using L1_L2_Pipe = ext::intel::pipe<class L1_L2_PipeClass, sibit_io,16>;
using L2_L3_Pipe = ext::intel::pipe<class L2_L3_PipeClass, sibit_io,16>;

class PER {
public:
    // constructor
    PER(){


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

        

        // start the FPGA autorun kernels
        MyAutorun_itm<Itm1, fixed_l1, fixed_root, ProducePipe, L1_L2_Pipe, 1, Lev1_Width>();
        MyAutorun_itm<Itm2, fixed_l2, fixed_l1, L1_L2_Pipe, L2_L3_Pipe, 2, Lev2_Width>();
        MyAutorun_lastlev<Itm3,fixed_l3, fixed_l2, L2_L3_Pipe, ConsumePipe1, ConsumePipe2, ConsumePipe3, Lev3_Width>();
        std::cout << "Constructor: Autorun started" << std::endl;
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
            typename InPipe, typename OutPipe1, typename OutPipe2, typename OutPipe3, int layersize>
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
                    }
                    OutPipe3::write(data_out_3);
                }
            });
        });
    }

    // CPU->FPGA
    // assume input data format is already handled at host (in_ptr is an array of sibit_io objects)
    // This is always for level 1, TLevDType = fixed_l1, layersize is the tree fanouts.
    template<typename OutPipe>
    // event Submit_Producer(queue &q, sibit_io *in_buf, size_t size) {
    event Submit_Producer(std::vector<sibit_io> &in_buf, size_t size) {
        // printf( "submit the producer\n");
        buffer in_buf_knl(in_buf);
        return q.submit([&](handler &h) [[intel::kernel_args_restrict]] {
            // host_ptr<sibit_io> in(in_buf);
            accessor in(in_buf_knl, h, read_only);
            h.single_task<P>([=] {
                for (size_t j = 0; j < size; j++) {
                    // uncomment for debugging
                    // PRINTF("producer, sample j=%d, sample input priority = %f\n",j,in[j].newx);
                    OutPipe::write(in[j]); 
                } 
            });      
        });
    }

    template <typename InPipe1, typename InPipe2, typename InPipe3, typename LastLevDType>
    event Submit_Consumer(std::vector<int> &out_buf1, std::vector<LastLevDType> &out_buf2,  std::vector<LastLevDType> &out_buf3, size_t size) {
        buffer out_buf1_knl(out_buf1);
        buffer out_buf2_knl(out_buf2);
        buffer out_buf3_knl(out_buf3);
        return q.submit([&](handler &h) [[intel::kernel_args_restrict]] {
            accessor out1(out_buf1_knl, h, write_only);
            accessor out2(out_buf2_knl, h, write_only);
            accessor out3(out_buf3_knl, h, write_only);
            h.single_task<C>([=] {
                for (int i = 0; i < size; i++) {
                    auto data1 = InPipe1::read();
                    out1[i] = data1;
                    // uncomment for debugging
                    // PRINTF("At consumer, sample %d, sampled index is %d\n",i,data1);
                    auto data2 = InPipe2::read();
                    out2[i] = data2;
                    auto data3 = InPipe3::read();
                    out3[i] = data3;
                }
            });
        });
    }

    void Init_Tree(fixed_root &root_pr){
    
        size_t batchsize = 2;       

        std::cout << "=== Running the kernel for initializing tree ===\n";

        std::vector<sibit_io> in;
        std::vector<int> sampled_idx; //sampling output
        std::vector<fixed_l3> out_pr_sampled; //sampling output
        std::vector<fixed_l3> out_pr_insertion; //insertion output

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

        root_pr=0;
        event p_e = Submit_Producer<ProducePipe>(in, batchsize);   
        event c_e = Submit_Consumer<ConsumePipe1, ConsumePipe2, ConsumePipe3, fixed_l3>(
                                                    sampled_idx,
                                                    out_pr_sampled,
                                                    out_pr_insertion,
                                                    batchsize);        
        p_e.wait();    // producer
        c_e.wait();   // consumer

        // events.wait();
        std::cout<<"out_pr_insertion for get priority:"<<out_pr_insertion[0]<<" "<<out_pr_insertion[1]<<"\n";

        printf("Init tree done. \n\n");
    }

    // void test(fixed_root &root_pr){
    //     root_pr = 1.8;
    //     printf("Test done. %f\n\n",root_pr);
    // }

    // Tin:sibit_io, Tout: fixed_l3
    // template <typename sibit_io, typename Tout>
    MultiKernel_out DoWorkMultiKernel(std::vector<sibit_io> &in, std::vector<int> &sampled_idx, std::vector<fixed_l3> &out_pr_sampled, std::vector<fixed_l3> &out_pr_insertion,
    // Tin* in, int* sampled_idx, Tout* out_pr_sampled, Tout* out_pr_insertion,
                            fixed_root &root_pr, 
                            size_t batchsize, size_t iterations) {
        // timing data
        // std::vector<double> latency_ms(iterations);
        // std::vector<double> process_time_ms(iterations);

        MultiKernel_out outp(batchsize);

        for (size_t i = 0; i < iterations; i++) {
    
            // update root stored on the host 
            for (size_t i = 0; i < batchsize; i++){
                if (in[i].update_flag==1){
                // std::cout<<"root_ptr = "<<root_pr<<" ";
                root_pr+=in[i].update_offset_array[0];
                }
            }
            // high_resolution_clock::time_point first_data_in, first_data_out;
            // auto start = high_resolution_clock::now();

            event p_e = Submit_Producer<ProducePipe>(in, batchsize);   
            event c_e = Submit_Consumer<ConsumePipe1, ConsumePipe2, ConsumePipe3, fixed_l3>(
                                                        sampled_idx,
                                                        out_pr_sampled,
                                                        out_pr_insertion,
                                                        batchsize);

            p_e.wait();    // producer
            c_e.wait();   // consumer

            // auto end = high_resolution_clock::now();
            // compute latency and processing time
            // duration<double, std::milli> latency = first_data_out - first_data_in;
            // duration<double, std::milli> process_time = end - start;
            // latency_ms[i] = latency.count();
            // process_time_ms[i] = process_time.count();

            
        }
        outp.sampled_idx = sampled_idx;
        outp.out_pr_sampled = out_pr_sampled;
        outp.out_pr_insertion =out_pr_insertion;
        outp.root_pr = root_pr;

        // // Validate the results
        // std::cout << "Sampled indices results: ";
        // for (int ii = 0; ii < batchsize; ++ii) {
        //     std::cout << sampled_idx[ii] << ' ';
        // }
        // std::cout << std::endl;

        // // 1st 16 elements should be: {0 9 13 62 49 30 36 19 60 1 8 16 32 27 40 31}.
        // std::cout << "Sampled values results: ";
        // for (int ii = 0; ii < batchsize; ++ii) {
        //     std::cout << out_pr_sampled[ii] << ' ';
        // }

        return outp;
    // compute and print timing information
    // PrintPerformanceInfo<T>("Multi-kernel",total_count, latency_ms, process_time_ms);
    }



private:
    sycl::queue q;
};