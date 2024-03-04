# PEARL
This is the implementation of RL using heterogeneous platform via DPC++ (sycl) and Torch.
The goal is to facilitate DRL application development with tools and familiar programming interfaces for DRL using heterogeneous platforms, while abstracting away the low-level hardware intricacies. 

The toolkit comprises three main components: a Host Runtime Code Template (run.py), a Parameterized Library of Accelerated Primitives (Libs_1API and Libs_Torch), and a System Composer (SysConfig).

<img src="https://github.com/CatherineMeng/HeteroRL/blob/main/images/PEARL_wrkflw.png" alt="drawing" width="380"/>

## Dependencies
This project uses conda to manage software dependencies.
Install all dependencies by creating a conda environment using:
```
conda env create -f install_env.yml
conda activate htroRLatari
```

On specific heterogeneous devices (e.g., FPGAs, integrated GPUs), the primitive implementations are pulled from sycl programs wrapped in Python interfaces. To utilize these features, install oneAPI and PyBind:
[Install oneAPI](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html)

[Install PyBind 11](https://pybind11.readthedocs.io/en/stable/installing.html)

## General Usage: System Composition

Edit the device List in **PerfPopulator.py** and run
```
python PerfPopulator.py
```
This will populate the PT.csv file used by the System Configurator. It stores the estimated/profiled latencies for the compute-primitives. 
Edit the BW.csv file to specify the bandwidth and latency of interconnections between devices. Note that the current runtime program support data transfer over PCIe between diffeernt devices, and assume DDR for communication among primitives within a CPU/GPU.
Now run
```
cd SysConfig
python SysCon.py
```
This will generate a mapping.json for runtime composition.

## General Usage: Runtime Deployment

### Step 0: Specify benchmarks, algorithm hyperparameters, memory configurations
Create a json file following the format with minimal parameters explained below:
```
{
    "alg": "<alg name, currently support DQN and DDPG>", 
    "batch_size_t": <training batch size (integer)>,
    "batch_size_i": <inference-insertion buffer size (integer)>,
    "replay_prioritized": "<bool>",
    "replay_size": <data storage memory size (integer)>,
    "replay_depth": <RM sum tree depth (integer)>,
    "replay_fanout": <RM sum tree fanout (integer)>,
    "env": "<Gym supported env name>",
    "hiddenl_sizes": <MLP policy hidden layer sizes (list of integers)>,
    "in_dim": <policy input state dimension (integer)>,
    "out_dim": <policy output action dimension (integer)>
}
```
See alp\_hp.json for an example.

### Step 1 (Optional): Compiling PY-SYCL Libraries
```
cd pybind/pysycl
icpx -shared -std=c++17 -fsycl -fPIC $(python3 -m pybind11 --includes) -I <path-to-pybind11>/pybind11/include bindings_replay.cpp -o sycl_rm_module$(python3-config --extension-suffix --includes)

cd ../pysyclfpga
dpcpp -O3 -Wall -shared -std=c++17 -fintelfpga -fPIC $(python3 -m pybind11 --includes) -I <path-to-pybind11>/pybind11/include bindings.cpp -o replay_module$(python3-config --extension-suffix --includes) -DFPGA_EMULATOR=0

cd ../..
```
This will generate shared object files in the corresponding directories which can be directly used by the runtime program.
For example, the second command generates a *replay_module.cpython-38-x86_64-linux-gnu.so* file in the pysyclfpga directory. If mapping to an FPGA, this file can be used in run.py via 
```
from pybind.pysyclfpga import replay_top
```

### Step 2 - Option 1: Automatic Deployment
This allows automatic device composition based on the output mapping from System Configuration process.
```
python run.py --mode auto
```
### Step 2 - Option 2: Manual Deployment
This allows plug-and-play of device composition for RL primitives. Directly copy and edit **custom_mapping.json** to select the devices, and run:
```
cp SysConfig/mapping.json ./custom_mapping.json
vim custom_mapping.json /*edit your custom device mappings*/
python run.py --mode manual
```

## Example Usage
The following example demonstrates (1) using the system composer on a hypothetical CPU-FPGA platform to generate a mapping result (mapping_out.json), and (2) runtime deployment of two DRL algorithms (DQN and DDPG) in two example benchmarks based on a mapping result (mapping.json) using a CPU-GPU platform.

### System Composer
Run the following to execute the script which invokes the System Composer: 
```
chmod +x run_sys_composer.sh 
./run_sys_composer.sh
```
The output should look something like
```
Profiled/Predicted performance data has been written to ./SysConfig/output_PT.csv

======== Step 1 Compute->Device ========
RM mapping: CPU 
Learner mapping: GPU1 
Theoretical peak throughput: 2226.086956521739 samples/sec
============== End Step 1 ==============

======== Step 2 Storage->Device ========
tot_traffic with Data Storage on CPU : 16001.006048000001
switching DS to CPU
tot_traffic with Data Storage on GPU1 : 32002.0010116459
tot_traffic with Data Storage on GPU2 : 64004.006528
tot_traffic with Data Storage on FPGA : 32004.004096
Data Storage mapping: CPU
============== End Step 2 ==============
```

### Runtime Program
The following are two minimal examples for running DQN in the CartPole environment and DDPG in the MountainCar environment.
The algorithm specifications and mapping specifications to be passed into the program are already created in the corresponding json files shown in the comman line arguments.

Example 1: Run DQN - CartPole
```
python run.py --algspec 'alg_hp.json' --mode manual --mappingspec 'custom_mapping1.json'
```
This program can be extended to other benchmarks with discrete action space.

Example 2: Run DDPG - MountainCar
```
python run_continuous.py --algspec 'alg_hp2.json' --mode manual --mappingspec 'custom_mapping2.json'
```
This program can be extended to other benchmarks with continuous action space.

Note: 
To test the runtime program on other algorithm/benchmarks, the entries "alg", "env", "hiddenl_sizes", "in_dim", and "out_dim" in alg_hp.json needs to be correctly modified.
To test the runtime program on reconfigurable platforms, do "Step 1 (Optional): Compiling PY-SYCL Libraries" above and make sure the design parameters in the SYCL header files algins correctly with the entries in alg_hp.json before running the runtime program. 

The outputs in the console should look something like 
```
=== actor 0 started
=== actor 2 started
=== actor 3 started
=== actor 1 started
Learner: Train Episodes finished, waiting for master
Effective throughput: 8900.22082046502 samples/second
***=== Actor 2 DONE TESTING
============REACHED MAIN BREAK============
Learner: Train Done
Terminate the actor processes and learner process
***=== Actor 0 DONE TESTING
Plotted for Actor2.png
Plotted for Actor0.png
learner_process_obj joined
***=== Actor 1 DONE TESTING
Plotted for Actor1.png
***=== Actor 3 DONE TESTING
Plotted for Actor3.png
actor_processes joined
```
The output plots should look something like (example: DQN-CartPole)

<img src="https://github.com/CatherineMeng/HeteroRL/blob/main/images/scores_actor_dqn.png" alt="drawing" width="190"/>