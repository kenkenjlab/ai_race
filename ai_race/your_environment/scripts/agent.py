#!python3
#-*- coding: utf-8 -*-

from brain import Brain


class Agent:
    def __init__(self, num_actions, batch_size, capacity, gamma):
        """課題の状態と行動の数を設定します"""
        self.brain = Brain(num_actions, batch_size, capacity, gamma)  # エージェントが行動を決定するための頭脳を生成

    def update_q_function(self):
        """Q関数を更新します"""
        self.brain.replay()

    def get_action(self, state, step):
        """行動の決定します"""
        action = self.brain.decide_action(state, step)
        return action

    def memorize(self, state, action, state_next, reward):
        """memoryオブジェクトに、state, action, state_next, rewardの内容を保存します"""
        self.brain.memory.push(state, action, state_next, reward)

    def save(self, path):
        self.brain.save(path)

    def load(self, path):
        self.brain.load(path)

    def update_target_network(self):
        self.brain.update_target_network()