import numpy as np
import gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

success_threashold=500
success_eps=5

def run_episode(env, parameters):
    observation = env.reset()
    observation = np.array(observation[0])
    totalreward = 0
    for _ in range(success_threashold):
        # print("obs array:",observation)
        # print("parameters:",parameters)
        action = 0 if np.matmul(parameters,np.array(observation)) < 0 else 1
        # print("actions:",action)
        # observation, reward, done, info = env.step(action)
        observation, reward, terminated, truncated, info = env.step(action)
        totalreward += reward
        if terminated:
            break
    return totalreward

# run
def test_policy(env, parameters):
    successs_count=0
    for _ in range(success_eps):
        observation = env.reset()
        observation = np.array(observation[0])
        totalreward = 0
        for _ in range(success_threashold):
            # print("obs array:",observation)
            # print("parameters:",parameters)
            action = 0 if np.matmul(parameters,np.array(observation)) < 0 else 1
            # print("actions:",action)
            # observation, reward, done, info = env.step(action)
            observation, reward, terminated, truncated, info = env.step(action)
            totalreward += reward
            if terminated:
                break
        if totalreward == success_threashold:
            successs_count+=1
    return successs_count/success_eps>0.8

if __name__ == "__main__":
    bestparams = None
    bestreward = 0
    for _ in range(10000):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env,parameters)
        if reward > bestreward:
            bestreward = reward
            print("best reward:",bestreward)
            bestparams = parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == success_threashold:
                print(test_policy(env,parameters))
                break