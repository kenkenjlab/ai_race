#!/usr/bin/env python

from base_learner import BaseLearner
from environment import Environment
import argparse

class LocalLearner(BaseLearner):

  def __init__(self, name = "untitled", model_output_dir = "./", online = False):
    BaseLearner.__init__(self, name, model_output_dir, online)
    self.__env = Environment(self.IMG_SIZE[0], self.IMG_SIZE[1], 3, one_side=True, online=online)

  def _get_action(self, img, stat = None):
    # Pass to environment
    if stat == None:
      action = self.__env.start_new_episode(img)
    else:
      action = self.__env.step_once(img, stat[0], stat[1])
    return True, 1.6, action

  def _finish_episode(self):
    self.__env.finish_episode()

  def _get_episode_count(self):
    return self.__env.get_episode_count()

  def _get_step_count(self):
    return self.__env.get_step_count()

  def _save_model(self):
    self.__env.save_model(self.model_output_dir, self.name)

  def _load_model(self, path):
    self.__env.load_model(path)

if __name__ == '__main__':
  import warnings
  warnings.filterwarnings("ignore", category=UserWarning)
  
  parser = argparse.ArgumentParser(description='Local learner')
  parser.add_argument('--online', help='Learn online or not. Default: false', default=False)
  parser.add_argument('--model_output_dir', help='Model output directory. Default: Current dirctory', default="./")
  parser.add_argument('--model_prefix', help='Model output file prefix. Default: "untitled"', default="untitled")
  args = parser.parse_args()

  print("Starting LocalLearner...")
  learner = LocalLearner(name = args.model_prefix, model_output_dir = args.model_output_dir, online = args.online)
  learner.run()
