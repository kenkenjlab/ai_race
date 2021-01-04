#!/usr/bin/env python

from base_learner import BaseLearner
from environment import Environment

class LocalLearner(BaseLearner):
  __env = Environment(3)

  def __init__(self):
    BaseLearner.__init__(self)

  def _get_action(self, img, stat = None):
    # Pass to environment
    if stat == None:
      action = self.__env.start_new_episode(img)
    else:
      action = self.__env.step_once(img, stat[0], stat[1])
    return action

  def _get_episode_count(self):
    return self.__env.get_episode_count()

  def _save_model(self):
    self.__env.save_model(self.model_output_dir, self.name)

  def _load_model(self, path):
    self.__env.save_model(path)

if __name__ == '__main__':
  print "Starting LocalLearner..."
  learner = LocalLearner()
  learner.run()
