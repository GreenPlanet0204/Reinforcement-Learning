# !pip install pyglet -U
# !pip install stable-baselines3
# !pip install gym[atari]
# !pip install ale-py
# !pip install autorom[accept-rom-license]

import gym
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import random
from datetime import datetime
from ale_py import ALEInterface
from ale_py.roms import MsPacman

ale = ALEInterface()
ale.loadROM(MsPacman)

"""Agent"""

class DQN_Agent:
  # 
  # Initializes attributes and constructs CNN model and target_model
  # 
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=5000)

    # Hyperparameters
    self.gamma = 1.0
    self.epsilon = 1.0
    self.epsilon_min = 0.1
    self.epsilon_decay = 0.995
    self.update_rate = 10000

    # Construct the DQN models
    self.model = self._build_model()
    self.target_model = self._build_model()
    self.target_model.set_weights(self.model.get_weights())
    self.model.summary()

  # 
  # Constructs CNN
  # 
  def _build_model(self):
    model = Sequential()

    model.add(Conv2D(32, (8, 8), strides=4, padding='same', activation='relu' ,input_shape=self.state_size))
    model.add(Conv2D(64, (4, 4), strides=2, padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
    return model

  # 
  # Stores experience in replay memory
  # 
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  # 
  # Choose action based on epsilon-greedy policy
  # 
  def act(self, state):
    # Random exploration
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0]) # Returns action using policy
  
  # 
  # Trains the model using randomly selected experiences in the replay memory
  # 
  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)

    for state, action, reward, next_state, done in minibatch:

      if not done:
        target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)))
      else:
        target = reward

      # Construct the target vector as follows:
      # 1. Use the current model to output the Q-value predictions
      target_f = self.model.predict(state)

      # 2. Rewrite the chosen action value with computed target
      target_f[0][action] = target

      # 3. Use vectors in the objective computations
      self.model.fit(state, target_f, epochs=1, verbose=0)
    
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  # 
  # Sets the target model parameters to the current model parameters
  # 
  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  # 
  # Loads a saved model
  # 
  def load(self, name):
    self.model.load_weights(name)

  # 
  # Saves parameters of a trained model
  # 
  def save(self, name):
    self.model.save_weights(name)

"""Preprocessing"""

def process_frame(frame):
  mspacman_color = np.array([210, 164, 74]).mean()
  img = frame[1:176:2, ::2] # Crop and downsize
  img = img.mean(axis=2) # Convert to greyscale
  img[img==mspacman_color] = 0 # Improve contrast by making pacman white
  img = (img - 128) / 128 - 1 # Normalize from -1 to 1

  return np.expand_dims(img.reshape(88, 80, 1), axis=0)

def blend_images(images, blend):
  avg_image = np.expand_dims(np.zeros((88, 80, 1), np.float64), axis=0)

  for image in images:
    avg_image += image
  
  if len(images) < blend:
    return avg_image / len(image)
  else:
    return avg_image / blend

"""Environment"""

env = gym.make("ALE/MsPacman-v5")
state_size = (88, 80, 1)
action_size = env.action_space.n
agent = DQN_Agent(state_size, action_size)

episodes = 500
batch_size = 8
skip_start = 90 # MsPacman-v0 waits for 90 actions before the episode begins
total_time = 0 # Counter for total number of steps taken
all_rewards = 0 # Used to compute avg reward over time
blend = 4 # Number of images to blend
done = False

for e in range(episodes):
  total_reward = 0
  game_score = 0
  state = process_frame(env.reset())
  images = deque(maxlen=blend) # Array of images to be blended
  images.append(state)

  for skip in range(skip_start): # skip the start of each game
    env.step(0)

  for time in range(20000):
    env.render(mode='rgb_array')
    total_time += 1

    # Every update_rate timesteps we update the target network parameters
    if total_time % agent.update_rate == 0:
      agent.update_target_model()

    # Return the avg of the last 4 frames
    state = blend_images(images, blend)

    # Transition Dynamics
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)

    game_score += reward
    reward -= 1  # Punish behavior which does not accumulate reward
    total_reward += reward

    # Return the avg of the last 4 frames
    next_state = process_frame(next_state)
    images.append(next_state)
    next_state = blend_images(images, blend)

    # Store sequence in replay memory
    agent.remember(state, action, reward, next_state, done)

    state = next_state

    if done:
      all_rewards += game_score

      print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}".
            format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), time, total_time))
      break

    if len(agent.memory) > batch_size:
      agent.replay(batch_size)

agent.save('models/5k-memory_1k-games')