import csv
import time
import random
import gym
import json
import sys
import torch
from SysConfig.FPGA_perfmodel import *

# Load the JSON content
with open('alg_hp.json') as f:
    hp = json.load(f)

alg = hp["alg"]
inf_batch_size = hp["batch_size_i"]
train_batch_size = hp["batch_size_t"]
replay_prioritized = hp["replay_prioritized"]
replay_size = hp["replay_size"]
replay_fanout = hp["replay_fanout"]
hidden_sizes = hp["hiddenl_sizes"]
in_dim=hp["in_dim"]
out_dim=hp["out_dim"]
env = gym.make(hp["env"])

# manually specify for continuous-space envs
policy_in_dim = 4
policy_out_dim = 2
# automatically specify for discrete-space envs
# policy_in_dim = env.observation_space.n
# policy_out_dim = env.action_space.n

# Prioritized RM on CPU, GPU, FPGA
# Uniform RM+DS on CPU, GPU
# from Libs_Torch.replay import ReplayMemory

# Learner on CPU, GPU, FPGA
if (alg=="DQN"):
    # sys.path.append('../../Libs_Torch')
    from Libs_Torch.dqn_learner import DQNLearner
elif (alg=="DDPG"):
    from Libs_Torch.ddpg_learner import DDPGLearner

# Function to generate (profile / predict) latency values for a given device
def pred_values(device):
    ret ={"RP-update":0,"RP-insert":0,"RP-sample":0,"LN":0,"RPLN-RP-update":0,"RPLN-RP-insert":0,"RPLN-RP-sample":0,"RPLN-LN":0}
    if (replay_prioritized=="True" and device == "igpu"): #integrated gpu
        from pybind.py_sycl.sycl_rm_module import SumTreeNary
        PTree = SumTreeNary(replay_size, replay_fanout)

        idx_vect=[0]*train_batch_size
        val_vect=[0]*train_batch_size
        for i in range(train_batch_size):
            idx_vect[i] = i
            val_vect[i] = 0.1*(i*4+i%4)
        start = time.perf_counter()
        PTree.set(idx_vect, val_vect); 
        ret["RP-update"] = (time.perf_counter()-start)* 1000 #ms
        
        
        idx_vect=[0]*inf_batch_size
        val_vect=[0]*inf_batch_size
        for i in range(inf_batch_size):
            idx_vect[i] = i
            val_vect[i] = 0.1*(i*4+i%4)
        start = time.perf_counter()
        PTree.set(idx_vect, val_vect); 
        ret["RP-insert"] = (time.perf_counter()-start)* 1000 #ms

        sampled_ind = [0] * train_batch_size
        sampling_values = [0] * train_batch_size
        for i in range(train_batch_size):
            sampling_values[i]=(random.random() * 65)  # Generate a random float from 0 to 65

        start = time.perf_counter()
        sampled_ind = PTree.get_prefix_sum_idx_sycl(sampling_values)
        ret["RP-sample"] = (time.perf_counter()-start)* 1000 #ms

    elif (replay_prioritized=="True" and device == "fpga"): #fpga
        # from pybind.py_sycl_fpga.replay_module import PER
        # from pybind.py_sycl_fpga.replay_module import sibit_io
        json_file_path = 'fpga_spec.json'
        with open(json_file_path, 'r') as json_file:
            fpga_spec = json.load(json_file)
        rm_perf = RM_perf( math.ceil(math.log(replay_size, replay_fanout)), replay_fanout, 128)
        ret["RP-sample"] = rm_perf['s-latency(ms)']
        ret["RP-insert"] = rm_perf['i-latency(ms)']
        ret["RP-update"] = rm_perf['u-latency(ms)']
        fit, l_perf = Learner_DSE([in_dim]+hidden_sizes+[out_dim], train_batch_size, fpga_spec['num_DSPs']-rm_perf['dsp_consp'], fpga_spec['num_SRAM_banks']-rm_perf['sram_consp'], fpga_spec['num_LogicRAM_banks']-rm_perf['logicram_consp'])
        ret["LN"] = l_perf

    elif (replay_prioritized=="True"): #cuda gpu, any cpu
        from Libs_Torch.replay import PrioritizedReplayMemory

        prmem = PrioritizedReplayMemory(replay_size, train_batch_size, inf_batch_size)

        tree_indices = torch.rand(train_batch_size)
        M = replay_size-1
        N = 2*replay_size-1
        tree_indices_scaled = (tree_indices * (N-M)+M).floor().long().numpy()
        td_errors = torch.rand(train_batch_size).numpy()
        start = time.perf_counter()
        prmem.update_through(td_errors)
        ret["RP-update"] = (time.perf_counter()-start)* 1000 #ms

        start = time.perf_counter()
        sbc = prmem.sample_through()
        ret["RP-sample"] = (time.perf_counter()-start)* 1000 #ms

        transition=torch.rand(train_batch_size)
        start = time.perf_counter()
        prmem.insert_through(transition, td_errors)
        ret["RP-insert"] = (time.perf_counter()-start)* 1000 #ms

    elif (replay_prioritized=="False"):
        from Libs_Torch.replay import ReplayMemory

    if (device != "fpga"):
        DLearner = DQNLearner(policy_in_dim, policy_out_dim, device)
        states = torch.rand(train_batch_size, policy_in_dim)
        actions = torch.rand(train_batch_size)
        next_states = torch.rand(train_batch_size, policy_in_dim)
        rewards = torch.rand(train_batch_size)
        dones = torch.zeros(train_batch_size).bool()
        start = time.perf_counter()
        DLearner.update_all_gradients(states, actions, next_states, rewards, dones, False)
        ret["LN"] = (time.perf_counter()-start)* 1000 #ms

    return list(ret.values())

# List of devices
# devices = ['cpu', 'cuda:0']
devices = ['cpu','fpga']
# devices = ['cpu', 'igpu', 'FPGA']
# devices = ['cpu', 'cuda:0', 'FPGA']

# CSV file name
csv_file = './SysConfig/output_PT.csv'

# Column names
columns = ['Device', 'RP-update', 'RP-insert', 'RP-sample', 'LN', 'RPLN-RP-update', 'RPLN-RP-insert', 'RPLN-RP-sample', 'RPLN-LN']

# Writing header to CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columns)

# Generating and writing hp for each device
for device in devices:
    # Generate values for the device
    values = pred_values(device)
    # print("values:",values)
    if (device=="cpu" or device[0:3]=="cuda" or device=="igpu"):
        for i in range(4,8): 
            values[i]=values[3]+values[i-4]
    else: #fpga, two modules are separate & latency-independent
        for i in range(4,8): 
            values[i]=values[i-4]


    # Writing the values to the CSV file
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([device] + values)

print(f"Profiled/Predicted performance data has been written to {csv_file}")
