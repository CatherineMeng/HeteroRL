'''
Policy objects are only put on learner process.
Replay is managed on master, can be offloaded to accelerator_replay
Learner has its separate thread, can be offloaded to a different accelerator than accelerator_replay
''' 

import time
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import numpy as np
# from torch.multiprocessing import Process, Pipe
from multiprocessing import Process, Pipe
# from replay import Memory
# from policy_dqn import DQN
from itertools import count
import sys
import argparse
import json

import matplotlib.pyplot as plt

import Libs_Torch.Config
import Config
# replay import
from Libs_Torch.replay import ReplayMemory
from Libs_Torch.replay import PrioritizedReplayMemory
from pybind.pysycl.sycl_rm_module import SumTreeNary # needs lib compilation (icpx)


# learner import 
from Libs_Torch.dqn_learner import DQNLearner, DQNN
from Libs_Torch.ddpg_learner import DDPGLearner, PolicyNN
# from pybind.pysycl.sycl_learner_module import DQNTrainer, params_out

parser = argparse.ArgumentParser(description="Runtime program description")
parser.add_argument("--mode", choices=["auto", "manual"], help="Set the system composition mode (auto or manual)", required=True)
parser.add_argument("--alg", choices=["DQN", "DDPG"], help="Set the target algorithm", required=False)

args = parser.parse_args()
mode = args.mode
alg = args.alg

# === Load the mapping JSON content == #
cpst_path = " "
if mode == "auto":
    cpst_path = "./SysConfig/mapping.json"
elif mode == "manual":
    cpst_path = "custom_mapping.json"
    # print("Running in manual mode")

with open(cpst_path, "r") as file:
    data = json.load(file)
ds_device = data["DS"]
rm_device = data["RM"]
learner_device = data["Learner"]
# use_gpu = use_cuda  # Set to True to enable training using CUDA if available
use_gpu = learner_device[0:3] == "GPU" and torch.cuda.is_available()

# === Load the alg hp JSON content === #
with open('alg_hp2.json') as f:
    hp = json.load(f)
alg = hp["alg"]
inf_batch_size = hp["batch_size_i"]
train_batch_size = hp["batch_size_t"]
replay_prioritized = hp["replay_prioritized"]
replay_size = hp["replay_size"]
replay_depth = hp["replay_depth"]
replay_fanout = hp["replay_fanout"]
hidden_sizes = hp["hiddenl_sizes"]
in_dim=hp["in_dim"]
out_dim=hp["out_dim"]
env_name = hp["env"]
lr = 1e-3

# === System Parallel Parameters === #
n_actors = 4
inf_batch_size = n_actors #Todo: generalize this (inf_batch_size for exp_buffer)
num_training_eps = 1000

use_cuda = learner_device[0:3] == "GPU" and torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Actor config.
epsilon = 0.5
epsilon_start = 0.9
epsilon_end = 0.05
# memory = Memory(10000)

# === learner object creation ===
def create_learner_actor(alg, learner_device):
    if learner_device[0:3] == "GPU" and torch.cuda.is_available():
        if alg == "DQN":
            return DQNLearner(in_dim, out_dim, 'cuda'), DQNN(in_dim, out_dim)
        elif alg == "DDPG":
            return DDPGLearner(in_dim, out_dim, 'cuda'), PolicyNN(in_dim, out_dim)
    elif learner_device[0:3] == "CPU":
        if alg == "DQN":
            # from Libs_Torch.dqn_learner import DQNLearner
            return DQNLearner(in_dim, out_dim, 'cpu'), DQNN(in_dim, out_dim)
        elif alg == "DDPG":
            # from Libs_Torch.ddpg_learner import DDPGLearner
            return DDPGLearner(in_dim, out_dim, 'cpu'), PolicyNN(in_dim, out_dim)
        # from pybind.py-sycl.sycl_learner_module import DQNTrainer
    elif (learner_device == "FPGA"):
        # Todo: pybind for sycl_learner_module_2l (current 3l)
        from pybind.pysycl.sycl_learner_module import DQNTrainer, params_out
        if alg == "DQN": 
            dplcy = DQNN(in_dim, out_dim)
            hw1, hb1, hw2, hb2 = dplcy.get_parameters_as_lists()
            return DQNTrainer(hw1, hb1, hw2, hb2)
        if alg == "DDPG":
            raise NotImplementedError("Implementing pybind for this function")

Learner, Policy_Net = create_learner_actor(alg, learner_device)

# === replay object creation ===
def create_replay(rm_device):
    if (rm_device == "CPU" or (rm_device == "GPU" and torch.cuda.is_available())):
    # from Libs_Torch.replay import PrioritizedReplayMemory
        if (replay_prioritized):
            return PrioritizedReplayMemory(replay_size, train_batch_size, inf_batch_size)
        else:
            return ReplayMemory(replay_size, train_batch_size, inf_batch_size)
    elif (rm_device == "GPU" and not torch.cuda.is_available()):
        from pybind.pysycl.sycl_rm_module import SumTreeNary
        from pybind.pysycl import replay_top
        return replay_top(fanout, train_batch_size, inf_batch_size, replay_size)
    # from pybind.py-sycl import SumTreeNary
    elif (rm_device == "FPGA"):
        from pybind.pysyclfpga import replay_top # needs lib compilation (dpcpp)
        return replay_top(fanout, replay_depth, train_batch_size, inf_batch_size, replay_size)

Replay_Memory = create_replay(rm_device)

# Define a function to run the actor process
# def actor_process(conn, actor_id, policy_net, memory, n_actions):
def actor_process(param_conn, actor_id, data_collection_conn):
    # if use_cuda and torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("USING GPU")
    # else:
    device = torch.device("cpu")
    env = gym.make(env_name)
    policy_net = Policy_Net.to(device)
    print("=== actor", actor_id,"started")
    p_update_cnt=0
    flag_break_outer=0
    avg_score_plot = [0]*(num_training_eps*2+1)
    last_score_plot = [0]*(num_training_eps*2+1)
    i=0 #count num episodes
    # while True:
    while not flag_break_outer:
        state = env.reset()[0]
        done = False
        # run one episode 
        rewards=0  
        i+=1
        epsilon = epsilon_end - epsilon_start * (i / num_training_eps) + epsilon_start
        # while not done:
        for t in count():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_net(state_tensor)
            # --- get action ---
            action = torch.argmax(action_probs).item()
            random_action = np.random.randint(0, 2)
            
            if (np.random.rand() < epsilon):
                action = random_action
            # --- end get action --- 
            next_state, reward, done, _, _ = env.step([action])
            rewards+=reward
            data_collection_conn.send([state, action, reward, next_state, done])
            state = next_state
            if (param_conn.poll()): # there is updated param to receive
                try: 
                    parameters = param_conn.recv()  # Receive policy network parameters from the learner
                except (FileNotFoundError):
                    # FileNotFoundError: [Errno 2] No such file or directory
                    print('Skipped param sync.') 
                except (ConnectionResetError):
                    # ConnectionResetError: [Errno 104] Connection reset by peer
                    print('Connection reset by peer.') 
                except (EOFError):
                    # raise EOFError
                    print('Pipe closed, EOFError.') 
                if parameters is not None:
                    # print("=== actor", actor_id,"=== received updated params from learner after Learner itr",p_update_cnt)
                    policy_net.load_state_dict(parameters)
                    
                    
            if done or t==200:
                episode_durations = t + 1
                avg_score_plot[i]=(avg_score_plot[i-1] * 0.99 + episode_durations * 0.01)
                last_score_plot[i]=(episode_durations)
                # train_time_epsavg_total+=1000*train_time_total/episode_durations #in ms
                p_update_cnt+=1
                break

        if (p_update_cnt == num_training_eps*2):
            print("***=== Actor", actor_id,"DONE TESTING")
            data_collection_conn.send("Adone")
            flag_break_outer=1

    plt.title('reward')
    
    plt.plot(last_score_plot, '-',label='last scores')
    plt.plot(avg_score_plot, 'r-',label='avg scores')
    plt.legend()
    plt.savefig("scores of Actor"+str(actor_id)+".png")
    print("Plotted for Actor"+str(actor_id)+".png")
    env.close()

# Define the function to run the learner process
def learner_process(param_conns, data_transfer_conn, batch_size, gamma, use_gpu):
    # print("Learner start")
    device = "cuda" if use_gpu else "cpu"
    learn_itr_cnt=0
    while True:
    # for _ in range(num_training_eps):
        learn_itr_cnt+=1
        if (learn_itr_cnt<=num_training_eps):
            transitions = data_transfer_conn.recv()
            # # print("Learner itr",learn_itr_cnt,"received data from master")
            batch = list(zip(*transitions))
            # # print("batch[0]:",batch[0])
            state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
            action_batch = torch.tensor(batch[1], dtype=torch.int64).to(device)
            reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
            next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(device)
            done_batch = torch.tensor(batch[4], dtype=torch.bool).to(device)
            # state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_transfer_conn.recv()
            # print("Learner itr",learn_itr_cnt,"received data from master")
            # print("here, itr",learn_itr_cnt)
            latency, new_prs, new_params = Learner.update_all_gradients(state_batch, action_batch, next_state_batch, reward_batch, done_batch, learn_itr_cnt%10==0)
            # print("Learner.update_all_gradients")
            Replay_Memory.update_through([new_prs]*train_batch_size) #todo: check consistency for tensor size returned in different algs

            # Send back the updated policy network parameters to the actors
            if (learn_itr_cnt%10 ==0):
                for param_conn in param_conns:
                    param_conn.send(new_params)
        
        else:
            print("Learner: Train Episodes finished, waiting for master")
            transitions = data_transfer_conn.recv()
            if (transitions=="Train done from master"):
                print("Learner: Train Done")
                break
            for param_conn in param_conns:
                param_conn.send(policy_net.state_dict())


# Define the main function to start the training process
def main():

    # Create pipes for communication between the master process and the actor processes
    data_collection_pipes = [Pipe() for _ in range(n_actors)]
    # Create pipes for communication between the learner process and the actor processes
    actor_pipes = [Pipe() for _ in range(n_actors)]
    # Create pipes for communication between the master process and the learner process
    data_transfer_pipe = Pipe()

    t_start = time.perf_counter()
    # Create and start actor processes
    actor_processes = [Process(target=actor_process, args=(actor_pipe[0], i, pipe_data[1]))
                    for i, (actor_pipe, pipe_data) in enumerate(zip(actor_pipes,data_collection_pipes))]
    for ap in actor_processes:
        ap.start()

    learner_conns = [conn[1] for conn in actor_pipes]
    # Create and start the learner process
    learner_process_obj = Process(target=learner_process, args=(learner_conns, data_transfer_pipe[0], train_batch_size, Config.gamma, use_gpu))
    learner_process_obj.start()

    exp_buffer =  [None]*n_actors
    # Training loop
    try:
        train_cnt=0
        n_actors_done = 0
        while (n_actors_done == 0):
            # replay insertion
            for i,pipe in enumerate(data_collection_pipes):
                sample = pipe[0].recv()
                if (sample=="Adone"):
                    n_actors_done+=1
                    data_transfer_pipe[1].send("Train done from master")
                    print("============REACHED MAIN BREAK============")
                    t_end = time.perf_counter()
                    print("total time for",num_training_eps,"training epiodes:",t_end-t_start)
                    # break 
                else: # replay insertion
                    exp_buffer[i] = sample
            new_prs=[0.1]*inf_batch_size #initial dp for insertion
            Replay_Memory.insert_through(exp_buffer,new_prs)

            if (Replay_Memory.get_current_size() >= 2*train_batch_size and train_cnt<num_training_eps):
                transitions = Replay_Memory.sample_through()
                data_transfer_pipe[1].send(transitions)
                train_cnt+=1

        # Terminate the actor processes and learner process
        print("Terminate the actor processes and learner process")
        for pipe in actor_pipes:
            pipe[0].close()
        data_transfer_pipe[1].send(None)
        data_transfer_pipe[0].close()

        learner_process_obj.join()
        print("learner_process_obj joined")

        for pipe in data_collection_pipes:
            # pipe[1].send(None)
            # pipe[1].send(None)
            while pipe[0].poll(timeout=1):
                pipe[0].recv()
            pipe[0].close()
        for ap in actor_processes:
            ap.join()
        # ap.terminate()
        print("actor_processes joined")

        

    except KeyboardInterrupt:
        # Terminate the actor processes and learner process if interrupte
        for ap in actor_processes:
            ap.join()

        learner_process_obj.join()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()