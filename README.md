# PEARL
This is the implementation of RL using heterogeneous platform via DPC++ and Torch.
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

On specific heterogeneous devices, the primitive implementations are pulled from sycl programs wrapped in Python interfaces. To utilize these features, install oneAPI and PyBind:
[Install oneAPI](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html)

[Install PyBind 11](https://pybind11.readthedocs.io/en/stable/installing.html)

## Usage: System Configuration

Edit the device List in **PerfPopulator.py** and run
```
python PerfPopulator.py
```
This will populate the PT.csv file used by the System Configurator. It has the estimated/profiled latencies for the compute-primitives. 
Edit the BW.csv file to specify the bandwidth and latency of interconnections between devices. Note that the current runtime program support data transfer over PCIe between diffeernt devices, and assume DDR for communication among primitives within a CPU/GPU.
Now run
```
cd SysConfig
python SysCon.py
```
This will generate a mapping.json for runtime composition.

## Usage: Runtime Deployment

### Step 1: Compiling PY-SYCL Libraries
```
cd pybind/pysycl
icpx -shared -std=c++17 -fsycl -fPIC $(python3 -m pybind11 --includes) -I <path-to-pybind11>/pybind11/include bindings_replay.cpp -o sycl_rm_module$(python3-config --extension-suffix --includes)

cd ../pysyclfpga
dpcpp -O3 -Wall -shared -std=c++17 -fintelfpga -fPIC $(python3 -m pybind11 --includes) -I <path-to-pybind11>/pybind11/include bindings.cpp -o replay_module$(python3-config --extension-suffix --includes) -DFPGA_EMULATOR=1

cd ../..
```
### Step 2 - Option 1: Automatic Deployment
This allows automatic device composition based on the output mapping from System Configuration process.
```
python run.py --mode auto --alg DQN
```
### Step 2 - Option 2: Manual Deployment
This allows plug-and-play of device composition for RL primitives. Directly copy and edit **custom_mapping.json** to select the devices, and run:
```
cp SysConfig/mapping.json ./custom_mapping.json
vim custom_mapping.json /*edit your custom device mappings*/
python run.py --mode manual --alg DQN
```