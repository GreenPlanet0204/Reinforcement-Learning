import gym

env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
for i_episode in range(10):
  observation = env.reset()
  for t in range(100):
    env.render()
    # print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
      print("{} timesteps taken for the episode".format(t+1))
      break

import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt    


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im


def save_random_agent_gif(env):
    frames = []
    for i in range(10):
        state = env.reset()        
        for t in range(1000):
            action = env.action_space.sample()

            frame = env.render(mode='rgb_array')
            frames.append(_label_with_episode_number(frame, episode_num=i))

            state, _, done, _ = env.step(action)
            if done:
                break

    env.close()

    imageio.mimwrite(os.path.join('./', 'random_agent.gif'), frames, fps=60)

env = gym.make('BipedalWalker-v3')
save_random_agent_gif(env)