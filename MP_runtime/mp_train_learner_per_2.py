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
from replay import ReplayMemory, SumTreenary, PrioritizedReplayMemory
from policy import PolicyNetwork



use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Parameters
n_actors = 4
capacity = 8192
batch_size = 16
gamma = 0.99
input_size = 4  # CartPole-v0 has an observation space of size 4
hidden_size = 8
output_size = 2  # Two possible actions: 0 (left) and 1 (right)
num_training_eps = 1000
lagging_itrs_target = 5 #number of lagged training iteration synchronizing target to policy

# Define a function to run the actor process
# def actor_process(conn, actor_id, policy_net, memory, n_actions):
def actor_process(param_conn, actor_id, data_collection_conn):
    env = gym.make('CartPole-v1')
    policy_net = PolicyNetwork(input_size, hidden_size, output_size)
    # use target net here for computing TD_error
    target_net = PolicyNetwork(input_size, hidden_size, output_size)
    target_net.load_state_dict(policy_net.state_dict())  # Initialize the target network with the policy network
    policy_update_cnt=0
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

            # Obtain the Q-values for the current state
            predicted_q_value = action_probs[action]
            # Calculate the target Q-value for the next state
            next_state_values = target_net(torch.tensor(next_state)).max().detach()
            # Compute the TD error
            td_error = abs((reward + gamma * next_state_values * (1 - done)) - predicted_q_value).detach()

            # data_collection_conn.send((state, action, reward, next_state, done))
            data_collection_conn.send(((state, action, reward, next_state, done), td_error))
            
            state = next_state

            if (param_conn.poll()): # there is updated param to receive
                policy_update_cnt+=1
                if (policy_update_cnt==lagging_itrs_target): #sync target net the same frequency as the learner does
                    policy_update_cnt=0
                    target_net.load_state_dict(policy_net.state_dict())

                try: 
                    parameters = param_conn.recv()  # Receive policy network parameters from the learner
                except (FileNotFoundError):
                    # FileNotFoundError: [Errno 2] No such file or directory
                    print('Skipped param sync.') 
                except (ConnectionResetError):
                    # ConnectionResetError: [Errno 104] Connection reset by peer
                    print('Learner has exited the learning loop. Connection reset by peer.') 
                except (EOFError):
                    # raise EOFError
                    print('Learner has exited the learning loop. Pipe closed.') 
                p_update_cnt+=1
                # print("=== actor", actor_id,"=== received updated params from learner after Learner itr",p_update_cnt)
                if parameters is not None:
                    policy_net.load_state_dict(parameters)
                    
            if (p_update_cnt == num_training_eps):
                print("***=== Actor", actor_id,"DONE TRAINING")
                # master_conn.send("Done") 
                flag_break_outer=1
                break

    env.close()

# Define the function to run the learner process
def learner_process(param_conns, data_collection_pipes, batch_size, gamma, use_gpu, master_conn):
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

    train_cnt=0
    n_actors_done = 0
    # Create the replay memory
    replay_memory = PrioritizedReplayMemory(capacity)
    flag_break_outer=0
    # while True:
    while not flag_break_outer:

        # transitions = data_transfer_conn.recv()

        # replay insertion
        for i,pipe in enumerate(data_collection_pipes):
            if (pipe.poll()): # there is new sample to receive
                try: 
                    sample = pipe.recv()
                    # print("Learner itr",train_cnt,"received data from actors")
  
                    replay_memory.push(sample[0],sample[1]) #sample[0] is the transition tuple, sample[1] is the td_error
                except (FileNotFoundError):
                    # FileNotFoundError: [Errno 2] No such file or directory
                    print('Skipped sample collection.') 
                except (ConnectionResetError):
                    # ConnectionResetError: [Errno 104] Connection reset by peer
                    print('Actor has exited the learning loop. Connection reset by peer.') 
                except (EOFError):
                    # raise EOFError
                    print('Actor has exited the learning loop. Pipe closed.') 

        if (replay_memory.get_current_size() >= batch_size and train_cnt<=num_training_eps):
            policy_net.to(device)
            # transitions = data_transfer_conn.recv()
            # print("Learner itr",train_cnt)
            transitions, indices = replay_memory.sample(batch_size)
            # data_transfer_pipe[1].send(transitions)
            train_cnt+=1
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

            # Update the priorities of experiences in the replay memory
            td_errors = torch.abs(expected_state_action_values - state_action_values.squeeze()).cpu().detach().numpy()
            replay_memory.update_priorities(indices, td_errors)

            if train_cnt % lagging_itrs_target == 0:  # Update the target network every 5 episodes
                target_net.load_state_dict(policy_net.state_dict())

            # Send back the updated policy network parameters to the actors
            for param_conn in param_conns:
                param_conn.send(policy_net.state_dict())
            # print("Learner itr",train_cnt,"sent updated params to actors")
        
        else:
            # if (n_actors_done>0):
            #     print("Learner: Train Done")
            #     break
            if (train_cnt>num_training_eps):
                print("Learner: Train Done")
                master_conn.send(1)
                flag_break_outer=1
                break
            for param_conn in param_conns:
                param_conn.send(policy_net.state_dict())


# Define the main function to start the training process
def main():


    use_gpu = use_cuda  # Set to True to enable training using CUDA if available


    # # Create pipes for communication between the master process and the actor processes
    # data_collection_pipes = [Pipe() for _ in range(n_actors)]
    # Create pipes for communication between the learner process and the actor processes
    learner_to_actor_pipes = [Pipe() for _ in range(n_actors)]
    actor_to_learner_pipes = [Pipe() for _ in range(n_actors)]
    done_signal_pipe = Pipe()
    # # Create pipes for communication between the master process and the learner process
    # data_transfer_pipe = Pipe()

    t_start = time.perf_counter()
    # Create and start actor processes
    actor_processes = [Process(target=actor_process, args=(actor_pipe[0], i, pipe_data[1]))
                    for i, (actor_pipe, pipe_data) in enumerate(zip(learner_to_actor_pipes,actor_to_learner_pipes))]
    for ap in actor_processes:
        ap.start()

    learner_to_actor_conns = [conn[1] for conn in learner_to_actor_pipes]
    actor_to_learner_conns = [conn[0] for conn in actor_to_learner_pipes]
    # Create and start the learner process
    learner_process_obj = Process(target=learner_process, args=(learner_to_actor_conns, actor_to_learner_conns, batch_size, gamma, use_gpu,done_signal_pipe[1]))
    learner_process_obj.start()

    while(1):
        try:

            # Terminate the actor processes and learner process
            # wait for all processes to finish
            if (done_signal_pipe[0].poll()):
                print("here")
                # done_signal=done_signal_pipe[0].recv()
                # if (done_signal==1):
                # print("done signal:",done_signal)
                learner_process_obj.join()
                for ap in actor_processes:
                    ap.kill()
                t_end = time.perf_counter()
                print("total time for",num_training_eps,"training epiodes:",t_end-t_start)
                break

        except KeyboardInterrupt:
            # Terminate the actor processes and learner process if interrupte
            for ap in actor_processes:
                ap.join()
            learner_process_obj.join()

if __name__ == '__main__':
    main()
