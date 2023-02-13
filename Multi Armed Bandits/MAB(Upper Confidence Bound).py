# git clone https://github.com/JKCooper2/gym-bandits.git
# cd gym-bandits
# pip install -e .

import gym_bandits
import gym
import numpy as np
import math

env = gym.make("BanditTenArmedGaussian-v0")
env.reset()
num_rounds = 20000
count = np.zeros(10)
sum_rewards = np.zeros(10)
Q = np.zeros(10)

def UCB(iters):
  ucb = np.zeros(10)
  if iters < 10:
    return i
  
  else:
    for arm in range(10):
      upper_bound = math.sqrt((2 * math.log(sum(count))) / count[arm])
      ucb[arm] = Q[arm] + upper_bound
    return (np.argmax(ucb))

for i in range(num_rounds):
  arm = UCB(i)
  observation, reward, done, info = env.step(arm)
  count[arm] += 1
  sum_rewards[arm] += reward
  Q[arm] = sum_rewards[arm] / count[arm]

print('The optimal arm is {}'.format(np.argmax(Q)))