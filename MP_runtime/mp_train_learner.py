'''
Policy objects are only put on learner process.
Replay is managed on master, can be offloaded to accelerator_replay
Learner has its separate thread, can be offloaded to a different accelerator than accelerator_replay
''' 

import gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import numpy as np
# from torch.multiprocessing import Process, Pipe
from multiprocessing import Process, Pipe
from replay import ReplayMemory
from policy import PolicyNetwork



use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Parameters
n_actors = 4
capacity = 10000
batch_size = 128
gamma = 0.99
input_size = 4  # CartPole-v0 has an observation space of size 4
hidden_size = 8
output_size = 2  # Two possible actions: 0 (left) and 1 (right)
num_training_eps = 10000

# Define a function to run the actor process
# def actor_process(conn, actor_id, policy_net, memory, n_actions):
def actor_process(param_conn, actor_id, data_collection_conn):
    env = gym.make('CartPole-v1')
    policy_net = PolicyNetwork(input_size, hidden_size, output_size)
    print("=== actor", actor_id,"started")
    p_update_cnt=0
    flag_break_outer=0
    # while True:
    while not flag_break_outer:
        state = env.reset()[0]
        done = False
        # run one episode 
        rewards=0  
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_net(state_tensor)
            action = torch.argmax(action_probs).item()
            next_state, reward, done, _, _ = env.step(action)
            rewards+=reward
            data_collection_conn.send((state, action, reward, next_state, done))
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
                p_update_cnt+=1
                # print("=== actor", actor_id,"=== received updated params from learner after Learner itr",p_update_cnt)
                if parameters is not None:
                    policy_net.load_state_dict(parameters)
                    
            if (p_update_cnt == num_training_eps):
                print("***=== Actor", actor_id,"DONE TRAINING")
                data_collection_conn.send("Train done from actor")
                flag_break_outer=1
                break

    env.close()

# Define the function to run the learner process
def learner_process(param_conns, data_transfer_conn, batch_size, gamma, use_gpu):
    # print("Learner start")
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("USING GPU")
    else:
        device = torch.device("cpu")

    # Create the policy network and the target network
    policy_net = PolicyNetwork(input_size, hidden_size, output_size)
    target_net = PolicyNetwork(input_size, hidden_size, output_size)
    target_net.load_state_dict(policy_net.state_dict())  # Initialize the target network with the policy network

    policy_net.to(device)
    target_net.to(device)

    optimizer = optim.Adam(policy_net.parameters())
    learn_itr_cnt=0
    while True:
    # for _ in range(num_training_eps):
        learn_itr_cnt+=1
        if (learn_itr_cnt<=num_training_eps):
            policy_net.to(device)
            transitions = data_transfer_conn.recv()
            # print("Learner itr",learn_itr_cnt,"received data from master")
            batch = list(zip(*transitions))
            state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
            action_batch = torch.tensor(batch[1], dtype=torch.int64).to(device)
            reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(device)
            next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(device)
            done_batch = torch.tensor(batch[4], dtype=torch.bool).to(device)

            state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
            next_state_values = target_net(next_state_batch).max(1)[0].detach()
            expected_state_action_values = (next_state_values * gamma) + reward_batch

            loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
            # print("loss:",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if learn_itr_cnt % 5 == 0:  # Update the target network every 5 episodes
                target_net.load_state_dict(policy_net.state_dict())

            # Send back the updated policy network parameters to the actors
            for param_conn in param_conns:
                param_conn.send(policy_net.state_dict())
            # print("Learner itr",learn_itr_cnt,"sent updated params to actors")
        
        else:
            transitions = data_transfer_conn.recv()
            if (transitions=="Train done from master"):
                print("Learner: Train Done")
                break
            for param_conn in param_conns:
                param_conn.send(policy_net.state_dict())


# Define the main function to start the training process
def main():


    use_gpu = use_cuda  # Set to True to enable training using CUDA if available

    # Create the replay memory
    replay_memory = ReplayMemory(capacity)

    # Create pipes for communication between the master process and the actor processes
    data_collection_pipes = [Pipe() for _ in range(n_actors)]
    # Create pipes for communication between the learner process and the actor processes
    actor_pipes = [Pipe() for _ in range(n_actors)]
    # Create pipes for communication between the master process and the learner process
    data_transfer_pipe = Pipe()

    # Create and start actor processes
    actor_processes = [Process(target=actor_process, args=(actor_pipe[0], i, pipe_data[1]))
                    for i, (actor_pipe, pipe_data) in enumerate(zip(actor_pipes,data_collection_pipes))]
    for ap in actor_processes:
        ap.start()

    learner_conns = [conn[1] for conn in actor_pipes]
    # Create and start the learner process
    learner_process_obj = Process(target=learner_process, args=(learner_conns, data_transfer_pipe[0], batch_size, gamma, use_gpu))
    learner_process_obj.start()

    # Training loop
    try:
        # for i in range(10000):  # You can adjust the number of training episodes here
        train_cnt=0
        n_actors_done = 0
        # while(1):
        while (n_actors_done==0):
            # replay insertion
            for i,pipe in enumerate(data_collection_pipes):
                sample = pipe[0].recv()
                if (sample=="Train done from actor"):
                    n_actors_done+=1
                    data_transfer_pipe[1].send("Train done from master")
                    print("============REACHED MAIN BREAK============")
                    # break 
                else:
                    replay_memory.push(sample)
            # print("MASTER: train_cnt", train_cnt, "Collecting from actors")

            # print("MASTER: train_cnt", train_cnt, "len(replay_memory):",len(replay_memory))
            if (len(replay_memory) >= batch_size and train_cnt<num_training_eps):
                transitions = replay_memory.sample(batch_size)
                data_transfer_pipe[1].send(transitions)
                train_cnt+=1
                # print("MASTER: train_cnt", train_cnt, "sent samples to learner")
  
            # if (n_actors_done==1):
            #     print("MASTER: Train done")
            #     break #as long as one of the actor is updated with the newest policy, we can exit training

        # Terminate the actor processes and learner process
        for pipe in actor_pipes:
            pipe[0].close()
        data_transfer_pipe[1].send(None)
        data_transfer_pipe[0].close()
        for pipe in data_collection_pipes:
            # pipe[1].send(None)
            pipe[0].close()

        learner_process_obj.join()
        for ap in actor_processes:
            ap.join()

        

    except KeyboardInterrupt:
        # Terminate the actor processes and learner process if interrupte
        for ap in actor_processes:
            ap.join()

        learner_process_obj.join()

if __name__ == '__main__':
    main()
