from replay_module import PER
from replay_module import sibit_io

# manually ensure the following is equivalent to those defined in replay_cpplib.cpp
K = 4 #fanout 
D = 4 #depth

# ========== TestBench: Init ===========
replay_manager = PER()

root_pr=0
# replay_manager.Test(root_pr)
replay_manager.Init(root_pr)

batchsize=16;
iterations = 1;
# batchsize*4 is the insertion batch size. batchsize is the sampling/update batch sizes



# ========== TestBench: Insertion -> get priority ===========
# generate the input data for insertion
#  get_pr_value

in_list = [None]*batchsize*4
out_sampled_idx = [None]*batchsize*4
out_sampled_value = [None]*batchsize*4
out_insertion_getPr_value = [None]*batchsize*4

for ii in range(batchsize*4):
    in_list[ii]=sibit_io()
    out_sampled_idx[ii]=0
    out_sampled_value[ii]=0
    out_insertion_getPr_value[ii]=0

for ii in range(batchsize*4):
    in_list[ii].sampling_flag=0
    in_list[ii].update_flag=0
    in_list[ii].get_priority_flag=1
    in_list[ii].init_flag=0
    in_list[ii].pr_idx=ii

print("=== Running the get-priority kernel ===")
replay_manager.DoWorkMultiKernel(in_list, out_sampled_idx, out_sampled_value, out_insertion_getPr_value, \
root_pr, batchsize*4, iterations)
# validate the results 
print("out_pr_insertion: ")
pstr=""
for ii in range(batchsize*4):
    pstr+= str(out_insertion_getPr_value[ii])+' '
    # print(out_insertion_getPr_value[ii]) #should be all 0 if static init is successful. 
print(pstr) #should be all 0 if static init is successful. 

# std::cout << "\n";

# ========== TestBench: Insertion -> Update ===========
for ii in range(batchsize*4):
    in_list[ii].sampling_flag=0
    in_list[ii].update_flag=1
    in_list[ii].get_priority_flag=0
    in_list[ii].init_flag=0
    in_list[ii].update_index_array[0]=0
    in_list[ii].update_index_array[1]=(ii/K)/K
    in_list[ii].update_index_array[2]=ii/K
    in_list[ii].update_index_array[3]=ii

    for iii in range(D):
        in_list[ii].update_offset_array[iii]=0.1
        # in_list[ii].set_upd_offset_index(iii,0.1)

print("=== Running the update kernel ===")
MultiKernel_out = replay_manager.DoWorkMultiKernel(in_list, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,\
root_pr, batchsize*4, iterations)

# validate the results 
print("Completed the update kernel")
print("Root value (updated): ", MultiKernel_out.root_pr) #should return 6.4.
# // On the FPGA side: PRINTF in the producer (lev1) should accumulates to 1.6 in the end. 


# ========== TestBench: Sampling ===========
# size should be chunks
tb_rand=[0.1, 1.0, 1.4, 6.3, 5.0, 3.1, 3.7, 2.0, 6.1, 0.2, 0.9, 1.7, 3.3, 2.8, 4.1, 3.2]
for ii in range(batchsize):
    in_list[ii].sampling_flag=1
    in_list[ii].update_flag=0
    in_list[ii].get_priority_flag=0
    in_list[ii].init_flag=0
    in_list[ii].start=0
    in_list[ii].newx=tb_rand[ii]
print("sampling inlist sampling values from python host:", [in_list[ii].newx for ii in range(batchsize)])
print("Running the Sampling kernel")
MultiKernel_out = replay_manager.DoWorkMultiKernel(in_list, out_sampled_idx, out_sampled_value, out_insertion_getPr_value,
root_pr, batchsize, iterations)

# // validate the results 
pstr=""
print("Sampled indices results:")
for ii in range(batchsize):
    pstr+= str(MultiKernel_out.sampled_idx[ii])+' '
print(pstr)

# 1st 16 elements should be: {0 9 13 62 49 30 36 19 60 1 8 16 32 27 40 31}.
pstr=""
print("Sampled values results:")
for ii in range(batchsize):
    pstr+= str(MultiKernel_out.out_pr_sampled[ii])+' '
print(pstr)

print("\n")