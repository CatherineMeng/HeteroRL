'''
Policy objects are only put on learner process.
LR: learner and replay host codes on the same thread.
used if replay and learner are put on the same accelerator device
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
from replay import Memory
from replay import ReplayMemory
from policy_dqn import DQN
from itertools import count

import matplotlib.pyplot as plt

use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

torch.set_num_threads(1)
# Parameters
n_actors = 4
capacity = 10000
batch_size = 128
target_update = 10
gamma = 0.9
epsilon = 0.5
epsilon_start = 0.9
epsilon_end = 0.05 
input_size = 4  # CartPole-v0 has an observation space of size 4
hidden_size = 64
output_size = 2  # Two possible actions: 0 (left) and 1 (right)
num_training_eps = 1000
x_plot_lim = int(num_training_eps*1.5)
lr = 1e-3
# memory = Memory(10000)

# Define a function to run the actor process
# def actor_process(conn, actor_id, policy_net, memory, n_actions):
def actor_process(param_conn, actor_id, data_collection_conn, signal_conn,event):

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("USING GPU")
    else:
        device = torch.device("cpu")
    env = gym.make('CartPole-v1')
    policy_net = DQN()
    print("=== actor", actor_id,"started")
    p_update_cnt=0
    flag_break_outer=0
    avg_score_plot = [0]*(x_plot_lim+1)
    last_score_plot = [0]*(x_plot_lim+1)
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
            next_state, reward, done, _, _ = env.step(action)
            rewards+=reward
            data_collection_conn.send((state, action, reward, next_state, done))
            state = next_state

            if (param_conn.poll()): # there is updated param to receive
                try: 
                    parameters = param_conn.recv()  # Receive policy network parameters from the learner
                    if parameters is not None:
                        # print("=== actor", actor_id,"=== received updated params from learner after Learner itr",p_update_cnt)
                        policy_net.load_state_dict(parameters)
                except (FileNotFoundError):
                    # FileNotFoundError: [Errno 2] No such file or directory
                    print('Skipped param sync.') 
                except (ConnectionResetError):
                    # ConnectionResetError: [Errno 104] Connection reset by peer
                    print('Connection reset by peer.') 
                except (EOFError):
                    # raise EOFError
                    print('Pipe closed, EOFError.') 

            if done or t==200:
                episode_durations = t + 1
                avg_score_plot[i]=(avg_score_plot[i-1] * 0.99 + episode_durations * 0.01)
                last_score_plot[i]=(episode_durations)
                # train_time_epsavg_total+=1000*train_time_total/episode_durations #in ms
                p_update_cnt+=1
                break

        if (p_update_cnt == x_plot_lim):
            print("***=== Actor", actor_id,"DONE TESTING") 
            signal_conn.send("Adone")
            flag_break_outer=1
        
    plt.title('reward')
    
    plt.plot(last_score_plot, '-',label='last scores')
    plt.plot(avg_score_plot, 'r-',label='avg scores')
    plt.legend()
    plt.savefig("scores of Actor"+str(actor_id)+".png")
    print("Plotted for Actor"+str(actor_id)+".png")
    env.close()
    event.set()
    time.sleep(1)

# Define the function to run the learner process
def learner_process(param_conns, data_collection_conns,  signal_conn, use_gpu):
    # print("Learner start")
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("USING GPU")
    else:
        device = torch.device("cpu")

    # Create the policy network and the target network
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())  # Initialize the target network with the policy network

    # optimizer = optim.Adam(policy_net.parameters())
    optimizer = optim.RMSprop(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    learn_itr_cnt=0

    # Create the replay memory
    replay_memory = ReplayMemory(capacity)

    while True: 
        learn_itr_cnt+=1 

        # replay insertion
        for i,pipe in enumerate(data_collection_conns):
            sample = pipe.recv()
            if (sample=="Adone"):
                continue
            else: # replay insertion
                replay_memory.push(sample)
        # Sampling and training
        if (len(replay_memory) >= batch_size and learn_itr_cnt<=num_training_eps):
            # sampling
            transitions = replay_memory.sample(batch_size)
            # print("Learner itr",learn_itr_cnt,"received data from master")
            batch = list(zip(*transitions))
            # print("batch[0]:",batch[0])
            state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
            action_batch = torch.tensor(batch[1], dtype=torch.int64).to(device)
            reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
            next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(device)
            done_batch = torch.tensor(batch[4], dtype=torch.bool).to(device)

            # state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
            state_action_values = policy_net(state_batch).gather(1, action_batch[:, None].long()).squeeze()
            next_state_values = target_net(next_state_batch).max(1)[0].detach()
            expected_state_action_values = (next_state_values * ( ~done_batch) * gamma) + reward_batch

            # loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
            loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)
            # print("loss:",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if learn_itr_cnt % target_update == 0:  # Update the target network every 10 episodes
                target_net.load_state_dict(policy_net.state_dict())

            # Send back the updated policy network parameters to the actors
            for param_conn in param_conns:
                param_conn.send(policy_net.state_dict())
            # print("Learner itr",learn_itr_cnt,"sent updated params to actors")
        
        elif (learn_itr_cnt>num_training_eps):
            # print("Learner: Train Episodes finished, waiting for master")
            if(learn_itr_cnt==num_training_eps+1):
                print("Learner: Train Episodes finished, waiting for master")
                for param_conn in param_conns:
                    param_conn.send(policy_net.state_dict())
            if(signal_conn.poll()):
                msg = signal_conn.recv()
                if (msg=="Train done from master"):
                    print("Learner: Train Done")
                    break




# Define the main function to start the training process
def main():


    use_gpu = use_cuda  # Set to True to enable training using CUDA if available
    # multiprocessing.set_start_method('fork')

    # Create pipes for communication between the master process and the actor processes
    data_collection_pipes = [Pipe() for _ in range(n_actors)]
    # Create pipes for communication between the learner process and the actor processes
    param_pipes = [Pipe() for _ in range(n_actors)]
    # Create pipes for signaling between the actor processes and the master process
    actor_signal_pipes = [Pipe() for _ in range(n_actors)]
    # Create pipe for signaling between the learner process and the master process
    learner_signal_pipe=Pipe()

    event = multiprocessing.Event()

    t_start = time.perf_counter()
    # Create and start actor processes
    actor_processes = [Process(target=actor_process, args=(param_pipe[0], i, data_pipe[1], signal_pipe[1],event))
                    for i, (param_pipe, data_pipe, signal_pipe) in enumerate(zip(param_pipes,data_collection_pipes,actor_signal_pipes))]
    for ap in actor_processes:
        ap.start()

    learner_sendparam_conns = [conn[1] for conn in param_pipes]
    learner_receivedata_conns = [conn[0] for conn in data_collection_pipes]
    # Create and start the learner process
    learner_process_obj = Process(target=learner_process, args=(learner_sendparam_conns, learner_receivedata_conns, learner_signal_pipe[0],use_gpu))
    learner_process_obj.start()

    try:

        n_actors_done = 0
        while (n_actors_done == 0):
            # for i,pipe in enumerate(actor_signal_pipes):
            #     if (pipe[0].poll()):
            #         sample = pipe[0].recv()
            #         if (sample=="Adone"):
            #             n_actors_done+=1
            #             learner_signal_pipe[1].send("Train done from master")
            #             print("============REACHED MAIN BREAK============")
            #             t_end = time.perf_counter()
            #             print("total time for",num_training_eps,"training epiodes:",t_end-t_start)
            if event.is_set():
                print("============REACHED MAIN BREAK============")
                t_end = time.perf_counter()
                print("total time for",num_training_eps,"training epiodes:",t_end-t_start)
                n_actors_done+=1

        # Terminate the actor processes and learner process
        print("Terminate the actor processes and learner process")

        for ap in actor_processes:
            ap.terminate()
        print("actor_processes joined")

        learner_process_obj.terminate()
        print("learner_process_obj joined")

        

        # for pipe in param_pipes:
        #     pipe[1].close()
        # for pipe in signal_pipes:
        #     pipe[1].close()

        # for pipe in data_collection_pipes:
        #     # pipe[1].send(None)
        #     # pipe[1].send(None)
        #     while pipe[1].poll(timeout=1):
        #         pipe[0].recv()
        #     pipe[1].close()
        print("Here")
        # ap.terminate()

    except KeyboardInterrupt:
        # Terminate the actor processes and learner process if interrupte
        for ap in actor_processes:
            ap.join()

        learner_process_obj.join()

if __name__ == '__main__':
    main()
