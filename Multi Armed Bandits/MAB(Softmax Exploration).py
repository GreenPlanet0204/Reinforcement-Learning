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

def softmax(tau):

  total = sum([math.exp(val/tau) for val in Q])
  probs = [math.exp(val/tau)/total for val in Q]

  threshold = np.random.random()
  cumulative_prob = 0.0
  for i in range(len(probs)):
    cumulative_prob += probs[i]
    if (cumulative_prob > threshold):
      return i
  return np.argmax(probs)

for i in range(num_rounds):
  arm = softmax(0.5)
  observation, reward, done, info = env.step(arm)
  count[arm] += 1
  sum_rewards[arm] += reward
  Q[arm] = sum_rewards[arm] / count[arm]
print('The optimal arm is {}'.format(np.argmax(Q)))