import csv
import argparse
import gym
import json

# Define the path to your input CSV file
PT_file_path = "PT.csv"
BW_file_path = "BW.csv"
output_js = "mapping.json"

#default
mapping = {
    'DS': 'CPU',
    'RM': 'CPU',
    'Learner': 'CPU'
}

parser = argparse.ArgumentParser(description='PyTorch DQN solution of CartPole-v0')

parser.add_argument('--train_batch_size', type=int, default=512)
parser.add_argument('--insert_batch_size', type=int, default=256)
parser.add_argument('--env_name', type=str, default="CartPole-v1")

cfg = parser.parse_args()

def print_inputPT(PTdict,str_in):
    # Print the data dictionary
    print(f"======== {str_in} Inputs ========")
    for label, inner_dict in PTdict.items():
        print(f"Device: {label}")
        for col_label, value in inner_dict.items():
            print(f"{col_label}: {value}")
        print("------")
    print(f"======= End {str_in} Inputs =======")

def compute_mapping(PTdict):
    T_itr=float('inf')
    mappings=("A","B")
    for rm_device in PTdict.keys():
        for lnr_device in PTdict.keys():
            T_itr_curr=0
            if (rm_device!=lnr_device):
                T_itr_curr=PTdict[rm_device]["RP-sample"]+max(PTdict[rm_device]["RP-update"]+PTdict[rm_device]["RP-insert"],PTdict[lnr_device]["LN"])
            else:
                T_itr_curr=PTdict[rm_device]["RPLN-RP-sample"]+max(PTdict[rm_device]["RPLN-RP-update"]+PTdict[rm_device]["RPLN-RP-insert"],PTdict[lnr_device]["RPLN-LN"])
            if T_itr_curr<T_itr:
                T_itr=T_itr_curr
                mappings=(rm_device,lnr_device)
    return T_itr,mappings

# helper function of DS_mapping: generate a legal & ordered devs string for indexing BW table
def getbwkey(dev1,dev2):
    if (dev1=="CPU"):
        return 1, dev1+","+dev2
    elif (dev2=="CPU"):
        return 1, dev2+","+dev1
    elif (dev1==dev2):
        return 1, dev1+","+dev2
    else: #two different accelerators, neither are CPU, 2-round communication goes through cpu
        return 2, "CPU,"+dev1, "CPU,"+dev2

def DS_mapping(BWdict,D_lnr,D_actr,D_rm,PTdict):
    assert(D_lnr in PTdict.keys())
    assert(D_actr in PTdict.keys())
    Dd=""
    min_traffic = float('inf')
    
    # Get the environment exp data sizes
    env = gym.make(cfg.env_name)
    # Get the state size (observation space dimension)
    state_size = 1
    for item in env.observation_space.shape:
        state_size *= item
    # state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    C_lnr=cfg.train_batch_size * (state_size*2+action_size+2 + 1) *4/10**9 #convert to GB
    C_actr=cfg.insert_batch_size * (state_size*2+action_size + 2) *4/10**9
    C_rm=cfg.train_batch_size

    for dev1 in PTdict.keys():

        tot_traffic=0 #total traffic from other devices (D_lnr,D_actr) to the device selected
        
        # Traffic with learner device
        k1 = getbwkey(dev1,D_lnr)
        if (k1[0]==1):
            tot_traffic += BWdict[k1[1]]["latency-ms"] + C_lnr/(BWdict[k1[1]]["bandwidth-GBpms"])
        else:
            tot_traffic += BWdict[k1[1]]["latency-ms"] + C_lnr/(BWdict[k1[1]]["bandwidth-GBpms"])
            tot_traffic += BWdict[k1[2]]["latency-ms"] + C_lnr/(BWdict[k1[2]]["bandwidth-GBpms"])
        # Traffic with actors device (CPU)
        k2 = getbwkey(dev1,D_actr)
        if (k2[0]==1):
            tot_traffic += BWdict[k2[1]]["latency-ms"] + C_actr/(BWdict[k2[1]]["bandwidth-GBpms"])
        else:
            tot_traffic += BWdict[k2[1]]["latency-ms"] + C_actr/(BWdict[k2[1]]["bandwidth-GBpms"])
            tot_traffic += BWdict[k2[2]]["latency-ms"] + C_actr/(BWdict[k2[2]]["bandwidth-GBpms"])
         # Traffic with RM device 
        k3 = getbwkey(dev1,D_rm)
        if (k3[0]==1):
            tot_traffic += BWdict[k3[1]]["latency-ms"] + C_rm/(BWdict[k3[1]]["bandwidth-GBpms"])
        else:
            tot_traffic += BWdict[k3[1]]["latency-ms"] + C_rm/(BWdict[k3[1]]["bandwidth-GBpms"])
            tot_traffic += BWdict[k3[2]]["latency-ms"] + C_rm/(BWdict[k3[2]]["bandwidth-GBpms"])  
        
        print("tot_traffic with Data Storage on",dev1,":",tot_traffic)
        if tot_traffic<min_traffic:
            min_traffic=tot_traffic
            Dd=dev1
            print("switching DS to",Dd)
            
    return Dd

    # for ind in BWdict.keys():
    #     dev1=ind.split(",")[0]
    #     dev2=ind.split(',')[1]
    #     # print(dev1,dev2)

if __name__ == "__main__":
    data_dict = {}  # Dictionary to store data
    with open(PT_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Read the header to get prim names
        header = next(csv_reader)
        prim_names = header[1:]  # Exclude the first column which contains labels pf primitive names

        # Iterate through devices
        for row in csv_reader:
            device_name = row[0]  # First element in the row is the device_name
            data_dict[device_name] = {}  
            prim_latency = row[1:]  # The rest of the elements are perf data
            
            for prim, value in zip(prim_names, prim_latency):
                # print(f"Label: {device_name}, Column: {prim}, Value: {value}")
                data_dict[device_name][prim] = float(value)  # Store the value as float
               
    # print_inputPT(data_dict,"PT")

    print("\n======== Step 1 Compute->Device ========")
    s1_results = compute_mapping(data_dict)
    print("RM mapping:",s1_results[1][0],"\nLearner mapping:",s1_results[1][1],"\nTheoretical peak throughput:",1000*cfg.train_batch_size/s1_results[0],"samples/sec")
    print("============== End Step 1 ==============")
    mapping['RM'] = s1_results[1][0]
    mapping['Learner'] = s1_results[1][1]

    bw_dict = {}  # Dictionary to store data
    with open(BW_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Read the header to get prim names
        header = next(csv_reader)
        cols = header[1:]  

        # Iterate through devices
        for row in csv_reader:
            devices_name = row[0]  # First element in the row 
            bw_dict[devices_name] = {}  
            vals = row[1:]  # The rest of the elements are perf data
            
            for col, value in zip(cols, vals):
                # print(f"Label: {device_name}, Column: {prim}, Value: {value}")
                bw_dict[devices_name][col] = float(value)  # Store the value as float

    # print_inputPT(bw_dict,"BW")
    
    print("\n======== Step 2 Storage->Device ========")
    s2_result = DS_mapping(bw_dict,s1_results[1][1],"CPU",s1_results[1][0],data_dict)
    print("Data Storage mapping:",s2_result)
    print("============== End Step 2 ==============")
    mapping['DS'] = s2_result


    with open(output_js, 'w') as json_file:
        json.dump(mapping, json_file)