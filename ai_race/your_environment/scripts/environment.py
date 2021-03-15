#!python3
#-*- coding: utf-8 -*-

from agent import Agent
import numpy as np
import torch
from PIL import Image as IMG
import torchvision.transforms as transforms
import cv2
import os
import math
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

class Environment:

  def __init__(self, width, height, num_actions, one_side = False, batch_size = 32, capacity = 10000, gamma = 0.99, target_update = 10, online=False):
    # Generate agent
    self.one_side = one_side
    if self.one_side:
      print('* One side mode')
      self.action_factor = float(num_actions - 1)
    else:
      print('* Both sides mode')
      self.action_factor = math.floor(float(num_actions) / 2)
    self.agent = Agent(width, height, num_actions, batch_size, capacity, gamma)
    self.episode_id = -1
    self.step_count = 0
    self.prev_state = None
    self.prev_action = None
    self.target_update = target_update
    self.online = online
    self.reward_sum = 0.0
    self.total_rewards = []

  def save_model(self, dir, prefix, suffix = ''):
    path = os.path.join(dir, "{}{:04}{}.pth".format(prefix, self.get_episode_count(), suffix))
    if not os.path.exists(dir):
      os.makedirs(dir)
    self.agent.save(path)

  def load_model(self, path):
    self.agent.load(path)

  def get_episode_count(self):
    return self.episode_id

  def get_step_count(self):
    return self.step_count

  def save_report(self, dir, prefix):
    # Save average rewards per episode as text
    csv_path = os.path.join(dir, "%s.csv" % prefix)
    with open(csv_path, "w") as f:
      text = "episode,total_reward\n"
      for i, r in enumerate(self.total_rewards):
        text += "%d,%f\n" % (i, r)
      f.write(text)

    # Draw and save graph of it
    png_path = os.path.join(dir, "%s.png" % prefix)
    x = np.array(range(self.episode_id + 1))
    y = np.array(self.total_rewards)
    fig = plt.figure()
    fig.suptitle("Total reward per episode")
    graph = fig.add_subplot(111, xlabel="Episode", ylabel="Total reward")
    graph.plot(x, y)
    #graph.set_yscale('log')
    graph.grid(axis='y', which='major',color='black',linestyle='-')
    graph.grid(axis='y', which='minor',color='black',linestyle='-')
    graph.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.savefig(png_path)
    plt.clf()
    plt.close(fig)

  def start_new_episode(self, observation):
    # Observation: Image BGR8
    #print("episode_id increment: {}+1={}".format(self.episode_id, self.episode_id+1))
    # Prepare new episode
    self.episode_id += 1
    self.step_count = 0
    self.reward_sum = 0.0

    # Prepare for upcoming next step
    state = self._cvt_to_tensor(observation)    # Regard observation as status 's' directly

    # Update target network
    if self.episode_id % self.target_update == 0:
        print('Updating target network')
        self.agent.update_target_network()

    # Get action
    action = self.agent.get_action(state, self.get_episode_count())

    # Prepare for upcoming next step
    self.prev_state = state
    self.prev_action = action
    self.step_count = 1

    return self._cvt_action(action)

  def step_once(self, observation, succeeded, failed, good, bad):
    # Observation: Image BGR8

    # Prepare for upcoming next step
    #if succeeded or failed:
    if failed:
      print('* Final round: (s,f)=({}, {})'.format(succeeded, failed))
      state = None
    else:
      if good:
        #print('* Better than before!')
        pass
      if bad:
        #print('* Worse than before!')
        pass
      state = self._cvt_to_tensor(observation)    # Regard observation as status 's' directly

    # Calculate reward
    reward = self._calc_reward(succeeded, failed, good, bad)

    # Record reward
    self.reward_sum += reward[0].item()

    # Memorize experience
    self.agent.memorize(self.prev_state, self.prev_action, state, reward)

    # Learn
    if self.online:
      self.agent.update_q_function()

    # Get next action
    if state == None:
      return 0.0    # TODO: Return flag which expresses the final round
    action = self.agent.get_action(state, self.get_episode_count())

    # Prepare for upcoming next step
    self.prev_state = state
    self.prev_action = action
    self.step_count += 1

    return self._cvt_action(action)

  def finish_episode(self):
    # Record average reward in this current episode
    if self.episode_id == len(self.total_rewards):
      self.total_rewards.append(self.reward_sum)
      #print("total_rewards.append: {}".format(len(self.total_rewards)))

    # Stop processing if online-learning mode
    if self.online:
      return

    # Replay if offline-learning mode
    print('Replaying {} times...'.format(self.step_count))
    time_begin_total = time.time()
    for i in range(self.step_count):
      time_begin = time.time()
      self.agent.update_q_function()
      time_diff = time.time() - time_begin
      print('[{}/{}] Replay finished. proc={:.3f}[s]'.format(i + 1, self.step_count, time_diff))
    time_diff_total = time.time() - time_begin_total
    print('Replay all done! proc={:.3f}[s]'.format(time_diff_total))

  def _cvt_to_tensor(self, observation):
    h, w, c = observation.shape
    img = IMG.fromarray(observation)
    img = transforms.ToTensor()(img)
    state = torch.zeros([1, c, h, w])
    state[0] = img
    return state

  def _calc_reward(self, succeeded, failed, good, bad):
    val = 0.0
    if succeeded:
      val = 1.0
    elif failed:
      val = -1.0
    elif good:
      val = 0.0
    elif bad:
      val = 0.0

    reward = torch.FloatTensor([val])
    return reward

  def _cvt_action(self, action_tensor):
    # Here action_tensor is torch.LongTensor of size 1x1
    val = float(action_tensor[0, 0])
    if self.one_side:
      action = val / self.action_factor
    else:
      action = (val - self.action_factor) / self.action_factor
    return action
