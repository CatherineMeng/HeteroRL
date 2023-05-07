## Sum Tree for Replay on FPGA

Producer Kernel (Submit_Producer_SiblingItr): {Host -> Device input streaming} + {Tree-level 1 processing} + {Producer Kernel -> Intermediate Device Kernel on-chip streaming}

Intermediate Device Kernel (Submit_Intermediate_SiblingItr, repeatly call as needed): {On-chip Input data streaming} + {Intermediate tree-level processing} + {On-chip Output data streaming}

Consumer Kernel (Submit_Comsumer_SiblingItr): {On-chip Input data streaming} + {Last Tree-level processing} + {Device -> Host output data streaming}

Note: This implementation does not support static SRAM array. Future releaase of "device global" can help realize static SRAM management.

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