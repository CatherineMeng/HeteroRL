## FPGA emu

```
make -f Makefile.fpga hw_emu
./rmm-buffers.fpga_emu
```

## FPGA hw

```
nohup make -f Makefile.fpga hw > mylog.txt & disown
./rmm-buffers-hwtest.fpga
```
