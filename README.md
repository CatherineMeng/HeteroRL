# RL_OneAPI
This is the implementation of RL using heterogeneous platform via DPC++ and Torch.


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

### Automatic Deployment
This allows automatic device composition based on the output mapping from System Configuration process.
```
python run.py --auto
```
### Manual Deployment
This allows plug-and-play of device composition for RL primitives. Directly copy and edit **custom_mapping.json** to select the devices, and run:
```
cp SysConfig/mapping.json ./custom_mapping.json
vim custom_mapping.json /*edit your custom device mappings*/
python run.py --manual
```