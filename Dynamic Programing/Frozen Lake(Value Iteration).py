import gym
import numpy as np

env = gym.make('FrozenLake-v1')

print(env.observation_space.n)
print(env.action_space.n)

def value_iteration(env, gamma = 1.0):
  value_table = np.zeros(env.observation_space.n)
  no_of_iterations = 10000
  threshold = 1e-20

  for i in range(no_of_iterations):
    updated_value_table = np.copy(value_table)

    for state in range(env.observation_space.n):
      Q_value = []

      for action in range(env.action_space.n):
        next_states_rewards = []

        for next_sr in env.P[state][action]:
          trans_prob, next_state, reward_prob, _ = next_sr
          next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

        Q_value.append(np.sum(next_states_rewards))
      value_table[state] = max(Q_value)
    if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
      print('Value-iteration converged at iteration# %d.' %(i+1))
      break
  return value_table

def extract_policy(value_table, gamma = 1.0):
  policy = np.zeros(env.observation_space.n)

  for state in range(env.observation_space.n):
    Q_table = np.zeros(env.action_space.n)

    for action in range(env.action_space.n):
      for next_sr in env.P[state][action]:
        trans_prob, next_state, reward_prob, _ = next_sr
        print(trans_prob, value_table[next_state], reward_prob)
        Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

    policy[state] = np.argmax(Q_table)
  return policy

optimal_value_function = value_iteration(env=env, gamma=1.0)
# print(optimal_value_function[0])
optimal_policy = extract_policy(optimal_value_function, gamma=1.0)

print(optimal_policy)