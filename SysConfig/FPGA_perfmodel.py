# import Libs_Torch.Config
import math

# Agilex AGF 014
# M20K (20,480 Bits): 7,110
# MLAB (640 Bits): 24,360
# Variable-precision digital signal processing (DSP) blocks: 4,510
# 18 x 19 multipliers 9120
num_SRAM_banks = 7110
bits_per_SRAM = 20480
num_LogicRAM_banks = 24360
bits_per_LogicRAM = 640
num_DSPs = 4510
num_DSPs_per_MultAdd = 1
freq = 250*10**6 #Hz
datawidth = 32
DDR_bandwdith = 77 #GB/s
DDR_latency = 16 #ns

in_dim = 16
out_dim = 4

def RM_perf(depth, fanout, bs):
    res = {'logicram_consp':0, 'sram_consp':0, 'dsp_consp':2*depth}
 
    # prioritize using LogicRAM for earilier levels, avoid underutil of larger SRAM banks
    for i in range(depth):
        lev_width = fanout**i
        if (lev_width*32 < bits_per_SRAM):
            res['logicram_consp'] += math.ceil( datawidth*2*lev_width / bits_per_LogicRAM)
        else:
            res['sram_consp'] += math.ceil( datawidth*2*lev_width / bits_per_SRAM)
    
    # if (op=='sample'):
    res['s-latency(ms)']=(fanout*(bs+depth)*2/freq)*1000 + (bs*6*datawidth/8)/(DDR_bandwdith*10**6) 
    # elif (op=='update'):
    res['u-latency(ms)']=(depth*2/freq)*1000 + (bs*6*datawidth/8)/(DDR_bandwdith*10**6) 
    # else:
    res['i-latency(ms)']=(depth*2/freq)*1000 + (bs*6*datawidth/8)/(DDR_bandwdith*10**6) 
    return res

hidden_sizes=[64,64]
layer_dims = [in_dim] + hidden_sizes + [out_dim]
def P_cycle(layer_dims): #producer
    return sum(layer_dims)

def FW_cycle(LL,NL,PL):
    return 2*(LL+NL)/PL

def OBJ_cycle():
    return 2*L3

def BW_cyle(LL,NL,PL):
    return 2*(LL+NL)/PL

def WA_cyle(LL,NL,PL):
    return 2*(LL+NL)/PL

def C_cycle():
    return L1+L2+L2+L3

def Train_Lat_total_ms(bs):
    #on_chip=(BW_cyle(L3,L2)+WA_cyle(L1,L2)+WA_cyle(L2,L3))*bs + (P_cycle()+FW_cycle(L1,L2)+FW_cycle(L2,L3)+OBJ_cycle())
    on_chip=max(BW_cyle(L3,L2,P2),WA_cyle(L1,L2,P2),WA_cyle(L2,L3,P2))*bs + (P_cycle()+FW_cycle(L1,L2,P2)+FW_cycle(L2,L3,P3)+OBJ_cycle())*bs/6
    return on_chip/200000+T_pcie

def nearest_p2_floor(number):
    result = 1
    while result * 2 < number:
        result *= 2
    return result

def Learner_DSE(layer_dims, bs, dsp_bound, sram_bound, lram_bound):
    prop_ops = {}
    prop_dsps = {}
    dpf={} #max data parallel factors in each layer (other than batch size)
    # loop thru layer props to get a list of ops
    i=0
    while(i<len(layer_dims)-1):
        prop_ops["FW"+str(i)] = bs*layer_dims[i]*layer_dims[i+1]
        dpf["FW"+str(i)] = layer_dims[i+1]
        # print("layer",i,"FW:",layer_dims[i],layer_dims[i+1])
        i+=1
    while(i>1):
        prop_ops["BW"+str(i-1)] = bs*layer_dims[i]*layer_dims[i-1]
        prop_ops["WA"+str(i-1)] = bs*layer_dims[i]*layer_dims[i-1]
        dpf["BW"+str(i-1)] = layer_dims[i-1]
        dpf["WA"+str(i-1)] = layer_dims[i-1]*layer_dims[i]
        # print("layer",i,"BW:",layer_dims[i],layer_dims[i-1])
        # print("layer",i,"WA:",layer_dims[i],layer_dims[i-1])
        i-=1
    prop_ops["WA"+str(0)] = bs*layer_dims[1]*layer_dims[0]
    dpf["WA"+str(0)] = layer_dims[1]*layer_dims[0]
    
    # use op list and dsp bound to decide layer-parallelism
    total_ops = sum(prop_ops.values())
    for k,v in prop_ops.items():
        prop_dsps[k]=min(dpf[k],nearest_p2_floor(dsp_bound*v//total_ops))

    l_lat = 0 #ms
    i=0
    while(i<len(layer_dims)-1):
        l_lat += prop_ops["FW"+str(i)]/prop_dsps["FW"+str(i)]
        i+=1
    while(i>1):
        l_lat += max(prop_ops["BW"+str(i-1)]/prop_dsps["BW"+str(i-1)], prop_ops["WA"+str(i-1)]/prop_dsps["WA"+str(i-1)])
        i-=1
    l_lat += prop_ops["WA"+str(0)]/dpf["WA"+str(0)]
    
    # use layer-parallelism to check memory (bufffer) usage
    avail_msize_Kbit =  lram_bound*bits_per_LogicRAM/1024 + sram_bound*bits_per_SRAM
    i=0
    buf_act = 0 #complete buffer
    buf_wt = 0 #tiled loading
    while(i<len(layer_dims)-1):
        buf_act += bs*layer_dims[i]*datawidth/1024
        buf_wt += layer_dims[i]*layer_dims[i+1]*datawidth/1024
        # print("layer",i,"FW:",layer_dims[i],layer_dims[i+1])
        i+=1

    if (buf_act+buf_wt>avail_msize_Kbit):
        return False, l_lat*1000/freq #to ms
    return True, l_lat*1000/freq #to ms

        
# rm_perf = RM_perf(4, 16, 128)
# print("rm_perf:",rm_perf)
# fit, l_perf = Learner_DSE(layer_dims, 128, num_DSPs-rm_perf['dsp_consp'], num_SRAM_banks-rm_perf['sram_consp'], num_LogicRAM_banks-rm_perf['logicram_consp'])
# print("l_perf",l_perf)