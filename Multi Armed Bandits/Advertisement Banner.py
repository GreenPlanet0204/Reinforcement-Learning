# import the necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame()
df['Banner_type_0'] = np.random.randint(0, 2, 100000)
df['Banner_type_1'] = np.random.randint(0, 2, 100000)
df['Banner_type_2'] = np.random.randint(0, 2, 100000)
df['Banner_type_3'] = np.random.randint(0, 2, 100000)
df['Banner_type_4'] = np.random.randint(0, 2, 100000)

num_banner = 5
no_of_iterations = 100000
banner_selected = []
count = np.zeros(num_banner)
Q = np.zeros(num_banner)
sum_rewards = np.zeros(num_banner)

# define an epsilon-greedy function

def epsilon_greedy(epsilon):

  random_value = np.random.random()
  choose_random = random_value < epsilon

  if choose_random:
    action = np.random.choice(num_banner)
  else:
    action = np.argmax(Q)

  return action

# Start playing with epsilon-greedy policy

for i in range(no_of_iterations):
  banner = epsilon_greedy(0.5)

  reward = df.values[i, banner]
  count[banner] += 1
  sum_rewards[banner] += reward
  Q[banner] = sum_rewards[banner] / count[banner]

  banner_selected.append(banner)

# Plot the results

sns.distplot(banner_selected)
