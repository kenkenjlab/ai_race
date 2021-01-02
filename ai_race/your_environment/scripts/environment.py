#!python3
#-*- coding: utf-8 -*-

from agent import Agent
import numpy as np
import torch
from PIL import Image as IMG
import torchvision.transforms as transforms
import cv2

class Environment:
 
  def __init__(self, batch_size = 32, capacity = 10000, gamma = 0.99):
    # Generate agent
    self.agent = Agent(batch_size, capacity, gamma)
    self.episode_id = -1
    self.step_count = 0
    self.prev_state = None
    self.prev_action = None
  
  def get_episode_count(self):
    return self.episode_id

  def spin_once(self, observation, stat):
    (succeeded, failed, reset) = stat
    if reset:
      action = self._start_new_episode(observation)
    else:
      action = self._step_once(observation, succeeded, failed)
    return action

 
  def _start_new_episode(self, observation):
    # Observation: Image BGR8

    # Prepare new episode
    self.episode_id += 1
    self.step_count = 0

    # Prepare for upcoming next step
    img = IMG.fromarray(observation)
    img = transforms.ToTensor()(img)
    state = img  # Regard observation as status 's' directly

    '''
    # Get action
    action = self.agent.get_action(state, self.step_count)

    # Prepare for upcoming next step
    self.prev_state = state
    self.prev_action = action
    self.step_count = 1

    return self._cvt_action(action)
    '''

    cv2.imshow("img", observation)
    cv2.waitKey(10)
    return 1.0

  def _step_once(self, observation, succeeded, failed):
    # Observation: Image BGR8

    # Prepare for upcoming next step
    img = IMG.fromarray(observation)
    img = transforms.ToTensor()(img)
    state = img  # Regard observation as status 's' directly

    # Calculate reward
    reward = self._calc_reward(succeeded, failed)

    # Memorize experience
    self.agent.memorize(self.prev_state, self.prev_action, state, reward)

    # Learn
    #self.agent.update_q_function()

    '''
    # Get next action
    action = self.agent.get_action(state, self.step_count)

    # Prepare for upcoming next step
    self.prev_state = state
    self.prev_action = action
    self.step_count += 1

    return self._cvt_action(action)
    '''

    cv2.imshow("img", observation)
    cv2.waitKey(10)
    return 1.0

  def _calc_reward(self, succeeded, failed):
    val = 0.0
    if succeeded:
      val = 1.0
    elif failed:
      val = -1.0

    reward = torch.FloatTensor([val])
    return reward
  
  def _cvt_action(self, action_tensor):
    val = action_tensor[0].item()
    return (val - 1.0)
