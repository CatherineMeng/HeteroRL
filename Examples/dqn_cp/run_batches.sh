nohup python dqn_mlp_cartpole.py --batch_size=16 > b16log_cpu.txt & disown
nohup python dqn_mlp_cartpole.py --batch_size=32 > b32log_cpu.txt & disown
nohup python dqn_mlp_cartpole.py --batch_size=64 > b64log_cpu.txt & disown
nohup python dqn_mlp_cartpole.py --batch_size=128 > b128log_cpu.txt & disown
nohup python dqn_mlp_cartpole.py --batch_size=256 > b256log_cpu.txt & disown