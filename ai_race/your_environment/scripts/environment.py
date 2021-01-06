#!python3
#-*- coding: utf-8 -*-

from agent import Agent
import numpy as np
import torch
from PIL import Image as IMG
import torchvision.transforms as transforms
import cv2
import os

class Environment:

  def __init__(self, num_actions, batch_size = 32, capacity = 1000, gamma = 0.99):
    # Generate agent
    self.agent = Agent(num_actions, batch_size, capacity, gamma)
    self.episode_id = -1
    self.step_count = 0
    self.prev_state = None
    self.prev_action = None

  def save_model(self, dir, prefix, suffix = ''):
    path = os.path.join(dir, "{}{:04}{}.pth".format(prefix, self.get_episode_count(), suffix))
    self.agent.save(path)

  def load_model(self, path):
    self.agent.load(path)

  def get_episode_count(self):
    return self.episode_id

  def start_new_episode(self, observation):
    # Observation: Image BGR8

    # Prepare new episode
    self.episode_id += 1
    self.step_count = 0

    # Prepare for upcoming next step
    state = self._cvt_to_tensor(observation)    # Regard observation as status 's' directly

    # Get action
    action = self.agent.get_action(state, self.get_episode_count())

    # Prepare for upcoming next step
    self.prev_state = state
    self.prev_action = action
    self.step_count = 1

    return self._cvt_action(action)

  def step_once(self, observation, succeeded, failed):
    # Observation: Image BGR8

    # Prepare for upcoming next step
    if succeeded or failed:
      print('* Final round: (s,f)=({}, {})'.format(succeeded, failed))
      state = None
    else:
      state = self._cvt_to_tensor(observation)    # Regard observation as status 's' directly

    # Calculate reward
    reward = self._calc_reward(succeeded, failed)

    # Memorize experience
    self.agent.memorize(self.prev_state, self.prev_action, state, reward)

    # Learn
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

  def _cvt_to_tensor(self, observation):
    h, w, c = observation.shape
    img = IMG.fromarray(observation)
    img = transforms.ToTensor()(img)
    state = torch.zeros([1, c, h, w])
    state[0] = img
    return state

  def _calc_reward(self, succeeded, failed):
    val = 0.0
    if succeeded:
      val = 1.0
    elif failed:
      val = -1.0

    reward = torch.FloatTensor([val])
    return reward

  def _cvt_action(self, action_tensor):
    # Here action_tensor is torch.LongTensor of size 1x1
    val = float(action_tensor[0, 0])
    return (val - 1.0)
