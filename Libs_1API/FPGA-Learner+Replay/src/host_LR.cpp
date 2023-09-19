#include <iostream>
#include <vector>
#include "LR_fpga_lib.cpp" 
#include <chrono>
#include <ctime> 
#include <random>


int main() {
    int replay_size=64;
    int train_bsize=16;
    int insert_bsize = 16;
    size_t iterations = 1;
    PER_LNR rm_learner(replay_size,train_bsize);

    fixed_root root_pr = 0;
    // # ========== TestBench: Init ===========
    rm_learner.Init_Tree();

    int batchsize = 16;
    

    // # ========== TestBench: 1 iteration ===========
    std::vector<sibit_io> in_list_u_s(train_bsize); //for update and sampling 
    std::vector<sibit_io> in_list_i(insert_bsize); //for insertion
    std::vector<experience> in_data(insert_bsize);

    // randomly init inputs
    for (int i = 0; i < train_bsize; ++i) {
        in_list_u_s[i] = sibit_io();
        in_list_u_s[i].sampling_flag = 0;
        in_list_u_s[i].update_flag = 1;
        in_list_u_s[i].init_flag = 0;
        in_list_u_s[i].update_index_array[0]=0;
        in_list_u_s[i].update_index_array[1]=(i/K)/K;
        in_list_u_s[i].update_index_array[2]=i/K;
        in_list_u_s[i].update_index_array[3]=i;
        // new pr update offsets: new pr - old pr
        for (size_t ii=0; ii<D; ii++) in_list_u_s[i].update_offset_array[ii] = i*0.1;
    }
    for (int i = 0; i < insert_bsize; ++i) {
        in_list_i[i] = sibit_io();
        in_list_i[i].sampling_flag = 0;
        in_list_i[i].update_flag = 1;
        in_list_i[i].init_flag = 0;
        in_list_i[i].update_index_array[0]=0;
        in_list_i[i].update_index_array[1]=((i+1)/K)/K;
        in_list_i[i].update_index_array[2]=(i+1)/K;
        in_list_i[i].update_index_array[3]=(i+1);
        // new pr update offsets: new pr - old pr
        for (size_t ii=0; ii<D; ii++) in_list_i[i].update_offset_array[ii] = (i+3)*0.1;

        for (size_t ii=0; ii<L1; ii++) in_data[i].state[ii]=ii*0.2;
        for (size_t ii=0; ii<L1; ii++) in_data[i].next_state[ii]=ii*0.3;
        in_data[i].action=0;
        in_data[i].reward=0.1;
        in_data[i].done=0;
        in_data[i].pr=1.6;
    }



    std::vector<W1Fmt> w1_buf(L2);         // Initialize w1_buf with L2 elements
    std::vector<W2Fmt> w2_buf(L3);         // Initialize w2_buf with L3 elements
    std::vector<float> bias1_buf(L2, 0.0); // Initialize bias1_buf with L1 elements, initialized to 0.0
    std::vector<float> bias2_buf(L3, 0.0); // Initialize bias2_buf with L2 elements, initialized to 0.0
    std::vector<W2TranspFmt> w2t_buf(L2);  // Initialize w2t_buf with L2 elements
    // Initialize w1_buf and w2_buf with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0); // Adjust the range as needed
    for (int i = 0; i < L2; ++i) {
        for (int j = 0; j < L1; ++j) {
            w1_buf[i].w[j] = dis(gen);
        }
    }
    for (int i = 0; i < L3; ++i) {
        for (int j = 0; j < L2; ++j) {
            w2_buf[i].w[j] = dis(gen);
        }
    }
    // Initialize w2t_buf s
    for (int i = 0; i < L2; ++i) {
        for (int j = 0; j < L3; ++j) {
            w2t_buf[i].w[j] = w2_buf[j].w[i];
        }
    }
    auto start = std::chrono::system_clock::now();
    std::cout << "=== Running the main kernel ===" << std::endl;
    MultiKernel_out_LR out1 = rm_learner.DoWorkMultiKernel_LR(in_list_u_s, in_list_i, in_data, 
                            w1_buf, w2_buf, bias1_buf, bias2_buf, w2t_buf, 
                            train_bsize, insert_bsize);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "RM and Learner with batch size" <<train_bsize<< "finished computation with " 
              << "elapsed time: " << elapsed_seconds.count() << "s"
              << std::endl;

    // Validate the results
    // std::cout << "out_pr_insertion: ";
    // for (int ii = 0; ii < batchsize * 4; ++ii) {
    //     std::cout << out_insertion_getPr_value[ii] << ' ';
    // }
    // std::cout << std::endl;


    return 0;
}






