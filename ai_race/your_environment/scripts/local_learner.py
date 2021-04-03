#!/usr/bin/env python

from base_learner import BaseLearner
from environment import Environment
import argparse
import os
import numpy as np

class LocalLearner(BaseLearner):

  def __init__(self, name = "untitled", model_output_dir = "./", memory_output_dir = "./", online = False, pretrained_model = "", previous_memory = ""):
    BaseLearner.__init__(self, name, model_output_dir, memory_output_dir, online)
    self.__env = Environment(self.IMG_SIZE[0], self.IMG_SIZE[1], 3, one_side=True, online=online)
    if pretrained_model:
      self._load_model(pretrained_model)
    if previous_memory:
      self._load_memory(previous_memory)

  def _save_report(self):
    print("Saving learning report...")
    if not os.path.exists(self.model_output_dir):
      os.makedirs(self.model_output_dir)
    self.__env.save_report(self.model_output_dir, "total_reward")

  def _get_action(self, img, stat = None):
    # Pass to environment
    if stat == None:
      action = self.__env.start_new_episode(img)
    else:
      action = self.__env.step_once(img, stat[0], stat[1], stat[3], stat[4])
    return True, 1.6, action

  def _finish_episode(self):
    self.__env.finish_episode()
    self._save_report()

  def _get_episode_count(self):
    return self.__env.get_episode_count()

  def _get_step_count(self):
    return self.__env.get_step_count()

  def _save_model(self):
    self.__env.save_model(self.model_output_dir, self.name)

  def _load_model(self, path):
    self.__env.load_model(path)

  def _save_memory(self):
    self.__env.save_memory(self.memory_output_dir, self.name)

  def _load_memory(self, path):
    self.__env.load_memory(path)

if __name__ == '__main__':
  import warnings
  warnings.filterwarnings("ignore", category=UserWarning)

  parser = argparse.ArgumentParser(description='Local learner')
  parser.add_argument('--online', help='Learn online or not. Default: false', default=False)
  parser.add_argument('--model_output_dir', help='Model output directory. Default: Current dirctory', default="./")
  parser.add_argument('--memory_output_dir', help='Memory output directory. Default: Current dirctory', default="./")
  parser.add_argument('--prefix', help='Output file prefix. Default: "untitled"', default="untitled")
  parser.add_argument('--pretrained_model', help='Pre-trained file path.')
  parser.add_argument('--previous_memory', help='Previous memory file path.')
  args = parser.parse_args()

  print("Starting LocalLearner...")
  learner = LocalLearner(name = args.prefix, model_output_dir = args.model_output_dir, memory_output_dir = args.memory_output_dir, online = args.online, pretrained_model = args.pretrained_model, previous_memory = args.previous_memory)
  learner.run()
