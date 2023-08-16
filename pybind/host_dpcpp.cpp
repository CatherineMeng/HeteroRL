#include <iostream>
#include <vector>
#include "replay_cpplib.cpp" // Replace "replay_module.h" with the appropriate header file for PER and sibit_io classes
#include <chrono>
#include <ctime> 

int main() {
    PER replay_manager;

    fixed_root root_pr = 0;
    // # ========== TestBench: Init ===========
    replay_manager.Init_Tree(root_pr);

    int batchsize = 16;
    int iterations = 1;

    // # ========== TestBench: Insertion -> get priority ===========
    std::vector<sibit_io> in_list(batchsize * 4);
    std::vector<int> out_sampled_idx(batchsize * 4);
    std::vector<fixed_l3> out_sampled_value(batchsize * 4);
    std::vector<fixed_l3> out_insertion_getPr_value(batchsize * 4);

    for (int ii = 0; ii < batchsize * 4; ++ii) {
        in_list[ii] = sibit_io();
        out_sampled_idx[ii] = 0;
        out_sampled_value[ii] = 0;
        out_insertion_getPr_value[ii] = 0;
    }

    for (int ii = 0; ii < batchsize * 4; ++ii) {
        in_list[ii].sampling_flag = 0;
        in_list[ii].update_flag = 0;
        in_list[ii].get_priority_flag = 1;
        in_list[ii].init_flag = 0;
        in_list[ii].pr_idx = ii;
    }

    auto start = std::chrono::system_clock::now();
    std::cout << "=== Running the get-priority kernel ===" << std::endl;
    replay_manager.DoWorkMultiKernel(in_list, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
                                     root_pr, batchsize * 4, iterations);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Insertion-get_pr with batch size" <<batchsize * 4<< "finished computation with " 
              << "elapsed time: " << elapsed_seconds.count() << "s"
              << std::endl;

    // Validate the results
    std::cout << "out_pr_insertion: ";
    for (int ii = 0; ii < batchsize * 4; ++ii) {
        std::cout << out_insertion_getPr_value[ii] << ' ';
    }
    std::cout << std::endl;

    // # ========== TestBench: Insertion -> Update ===========
    for (int ii = 0; ii < batchsize * 4; ++ii) {
        in_list[ii].sampling_flag = 0;
        in_list[ii].update_flag = 1;
        in_list[ii].get_priority_flag = 0;
        in_list[ii].init_flag = 0;
        // in_list[ii].set_upd_input_index(0, 0);
        // in_list[ii].set_upd_input_index(1, (ii / K) / K);
        // in_list[ii].set_upd_input_index(2, ii / K);
        // in_list[ii].set_upd_input_index(3, ii);
        in_list[ii].update_index_array[0]=0;
        in_list[ii].update_index_array[1]=(ii / K) / K;
        in_list[ii].update_index_array[2]=ii / K;
        in_list[ii].update_index_array[3]=ii;

        for (int iii = 0; iii < D; ++iii) {
            // in_list[ii].set_upd_offset_index(iii, 0.1);
            in_list[ii].update_offset_array[iii]=0.1;
                
        }
    }

    start = std::chrono::system_clock::now();
    std::cout << "=== Running the update kernel ===" << std::endl;
    MultiKernel_out out_obj = replay_manager.DoWorkMultiKernel(in_list, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
                                                      root_pr, batchsize * 4, iterations);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "Update with batch size" <<batchsize * 4<< "finished computation with " 
              << "elapsed time: " << elapsed_seconds.count() << "s"
              << std::endl;

    // Validate the results
    std::cout << "Completed the update kernel" << std::endl;
    std::cout << "Root value (updated): " << out_obj.root_pr << std::endl; // should return 6.4.


    // ========== TestBench: Sampling ===========
    std::vector<double> tb_rand = {0.1, 1.0, 1.4, 6.3, 5.0, 3.1, 3.7, 2.0, 6.1, 0.2, 0.9, 1.7, 3.3, 2.8, 4.1, 3.2};

    for (int ii = 0; ii < batchsize; ++ii) {
        in_list[ii].sampling_flag = 1;
        in_list[ii].update_flag = 0;
        in_list[ii].get_priority_flag = 0;
        in_list[ii].init_flag = 0;
        in_list[ii].start = 0;
        in_list[ii].newx = tb_rand[ii];
    }

    start = std::chrono::system_clock::now();
    std::cout << "Running the Sampling kernel" << std::endl;
    out_obj = replay_manager.DoWorkMultiKernel(in_list, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
                                                      root_pr, batchsize, iterations);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "Sampling with batch size" <<batchsize<< "finished computation with " 
              << "elapsed time: " << elapsed_seconds.count() << "s"
              << std::endl;

    // Validate the results
    std::cout << "Sampled indices results: ";
    for (int ii = 0; ii < batchsize; ++ii) {
        std::cout << out_obj.sampled_idx[ii] << ' ';
    }
    std::cout << std::endl;

    // 1st 16 elements should be: {0 9 13 62 49 30 36 19 60 1 8 16 32 27 40 31}.
    std::cout << "Sampled values results: ";
    for (int ii = 0; ii < batchsize; ++ii) {
        std::cout << out_obj.out_pr_sampled[ii] << ' ';
    }
    std::cout << std::endl;
    return 0;
}






