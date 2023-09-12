
### KALU
Example 1: CartPole DQN, replay on CPU master thread, N actor threads and 1 learner thread
```
$conda activate htroRL

# with reward plots, one for each actor
$python mp_train_AL.py 

# no plots, uniform replay
$python mp_train_learner.py 

# no plots, PER
$python mp_train_learner_per_2.py 
```

Example 2: CartPole DQN, replay on CPU master thread, N actor threads and 1 learner thread
```
# with reward plot, one for the earlist finishing actor
$python mp_train_LR.py
```

serial baseline:
```
python serial_baseline_dqn_mlp.py
```