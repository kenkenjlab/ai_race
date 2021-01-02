#!/usr/bin/env python

from base_learner import BaseLearner
from environment import Environment

class LocalLearner(BaseLearner):
  __env = Environment()

  def __init__(self):
    #super().__init__()
    pass

  def _get_action(self, img, stat):
    # Pass to environment
    action = self.__env.spin_once(img, stat)
    return action
  
  def _get_episode_count(self):
    return self.__env.get_episode_count()

if __name__ == '__main__':
  print "Starting LocalLearner..."
  learner = LocalLearner()
  learner.run()
