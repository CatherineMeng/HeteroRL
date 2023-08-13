
# Test: Cartpole, DQN, MLP
### Train function corretly tested in gym env

## Creating conda env for Atari game tests
```
conda create --name htroRLatari python=3.8
conda activate htroRLatari
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install gym
pip install gym[atari,accept-rom-license]
pip install opencv-python
pip install matplotlib
```
## CartPole, MLP, DQN
### run once
```
conda activate htroRLatari
python dqn_mlp_cartpole.py
nohup python dqn_mlp_cartpole.py > mylog.txt & disown
```
### run for different batch sizes and collect outputs
```
chmod +x run_batches.sh
./run_batches.sh
```

## CartPole, MLP, DQN

## Pong, CNN, DQN

## Pong, CNN, DDPG
### run once
```
conda activate htroRLatari
cd ddpg_pong
nohup python ddpg_pong.py > mylog.txt & disown
```
### run for different batch sizes and collect outputs
```

```