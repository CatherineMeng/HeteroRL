
# Test: 2 algorithms, 2 models - 4 benchmarks
Train function & reward plots corretly tested in gym env

## Creating conda env for Atari game tests
```
conda create --name htroRLatari python=3.8
conda activate htroRLatari
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install gym
pip install gym[atari,accept-rom-license]
pip install opencv-python
pip install matplotlib
pip install tensorboard
pip install swig
pip install gym[box2d]
```
## CartPole, MLP, DQN
### run once
```
conda activate htroRLatari
cd dqn_cp
python dqn_mlp_cartpole.py
nohup python dqn_mlp_cartpole.py > mylog.txt & disown
```
### run for different batch sizes and collect outputs
```
chmod +x run_batches.sh
./run_batches.sh
```

## Pong, CNN, DQN
```
conda activate htroRLatari
cd dqn_pong
python dqn_pong.py
nohup python dqn_pong.py > mylog.txt & disown
```

## LunarLander, MLP, DDPG
```
conda activate htroRLatari
cd ddpg_lunarlander
python ddpg_ll.py
nohup python ddpg_ll.py > mylog.txt & disown
```

## MountCar, MLP, DDPG

```
conda activate htroRLatari
cd ddpg_mountcar
python ddpg_mountcar.py
nohup python ddpg_mountcar.py > mylog.txt & disown
```
