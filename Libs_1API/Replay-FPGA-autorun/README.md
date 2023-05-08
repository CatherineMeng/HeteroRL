## Sum Tree for Replay on FPGA

Note: To support static SRAM management without future releaase of "device global", build kernels that process tree levels with one-time one-chip Tree initialization outside of a while(1) loop.  This also gets rid of the "Init" component of the request structs communicated through the kernels. Put all streaming requests processing inside the while(1) loop.
The producer and consumer kernels are separated out and they are now only responsible for host<->device data streaming. The (D-1)-layer tree processing kernels are all auto-run.

Producer Kernel (Submit_Producer_SiblingItr): {Host -> Device input streaming} 

Intermediate Device Kernel (Submit_Intermediate_SiblingItr, repeatly call as needed - layer 1,2,3...): {On-chip Input data streaming} + {Intermediate tree-level processing} + {On-chip Output data streaming}

Consumer Kernel (Submit_Comsumer_SiblingItr): {Device -> Host output data streaming}

### FPGA Emulation

DevCloud, May 4

```
$ qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
icpx -g -fsycl -std=c++17 -I. -fintelfpga -DFPGA_EMULATOR host.cpp -o fpga_emu
```

Re-organized with src folder and cmake:
```
$ qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
mkdir build
cd build
cmake ..
```
software emulation: 
```
make fpga_emu
```

### FPGA Report

Re-organized with src folder and cmake:
```
make report
```
### FPGA Hardware Simulation

Re-organized with src folder and cmake:
```
make fpga_sim
```
### FPGA Hardware

Re-organized with src folder and cmake:
```
make fpga
```