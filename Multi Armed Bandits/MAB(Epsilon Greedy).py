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

def epsilon_greedy(epsilon):
  rand = np.random.random()
  if rand < epsilon:
    action = env.action_space.sample()
  else:
    action = np.argmax(Q)
  return action

for i in range(num_rounds):
  arm = epsilon_greedy(0.5)
  observation, reward, done, info = env.step(arm)
  count[arm] += 1
  sum_rewards[arm] += reward
  Q[arm] = sum_rewards[arm] / count[arm]

print('The optimal arm is {}'.format(np.argmax(Q)))