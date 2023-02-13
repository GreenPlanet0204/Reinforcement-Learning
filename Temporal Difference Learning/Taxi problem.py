import gym
import random

env = gym.make("Taxi-v3")

alpha = 0.4
gamma = 0.999
epsilon = 0.017

q = {}
for s in range(env.observation_space.n):
  for a in range(env.action_space.n):
    q[(s, a)] = 0.0

def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
  qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
  q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])

def epsilon_greedy_policy(state, epsilon):
  if random.uniform(0, 1) < epsilon:
    return env.action_space.sample()
  else:
    return max(list(range(env.action_space.n)), key = lambda x: q[(state, x)])

# For each episode
for i in range(8000):
  r = 0
  # first we initialize the environment

  prev_state = env.reset()
  while True:

    # In each state we select action by epsilon greedy policy
    action = epsilon_greedy_policy(prev_state, epsilon)

    # Then we take the selected action and move to the next state
    nextstate, reward, done, _ = env.step(action)

    # And we update the q value using the update_q_table() function
    # which updates q table according to our update rule
    update_q_table(prev_state, action, reward, nextstate, alpha, gamma)

    # Then we update the previous state as next state
    prev_state = nextstate

    # And store the rewards in r
    r += reward

    # If done i.e if we reached the terminal state of the episode
    # If break the loop and start the next episode
    if done:
      break
  
  print("total reward: ", r)
env.close()