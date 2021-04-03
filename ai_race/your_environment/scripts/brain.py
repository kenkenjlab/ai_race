#!python3
#-*- coding: utf-8 -*-

import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from transition import Transition
from replaymemory import ReplayMemory
from permemory import PERMemory
#import torchvision.models as models
from ateamnet import ATeamNet
import pickle

#------------------------------------------------

class Brain:
    TARGET_UPDATE = 10
    def __init__(self, width, height, num_actions, batch_size = 32, capacity = 10000, gamma = 0.99, prioritized = True, boltzmann = False):
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_actions = num_actions
        self.prioritized = prioritized
        self.boltzmann = boltzmann
        if self.boltzmann:
            print('* Boltzman Selection Mode')
        else:
            print('* Epsilon Greedy Mode')

        # 経験を記憶するメモリオブジェクトを生成
        if self.prioritized:
            print('* Prioritized Experience Replay Mode')
            self.memory = PERMemory(capacity)
        else:
            print('* Random Experience Replay Mode')
            self.memory = ReplayMemory(capacity)

        # Build network
        input_size = width * height
        self.policy_net = ATeamNet(input_size, self.num_actions)
        self.target_net = ATeamNet(input_size, self.num_actions)
        self.target_net.eval()

        # Set device type; GPU or CPU (Use GPU if available)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

        print('using device:', self.device)
        #print(self.policy_net)  # ネットワークの形を出力

        # 最適化手法の設定
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)

    def replay(self):
        """Experience Replayでネットワークの重みを学習 """

        # メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < self.batch_size:
            return

        # メモリからミニバッチ分のデータを取り出す
        transitions, indexes = self.memory.sample(self.batch_size)

        # ミニバッチの作成-----------------

        # transitionsは1stepごとの(state, action, state_next, reward)が、self.batch_size分格納されている
        # つまり、(state, action, state_next, reward)×self.batch_size
        # これをミニバッチにしたい。つまり
        # (state×self.batch_size, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        # バッチから状態、行動、報酬を格納（non_finalはdoneになっていないstate）
        # catはConcatenates（結合）のことです。
        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がself.batch_size分並んでいるのですが、
        # それを size self.batch_sizex4 に変換します
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]))

        # Set device type; GPU or CPU
        state_batch = state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        non_final_next_states = non_final_next_states.to(self.device)

        # ミニバッチの作成終了------------------

        # ネットワークを推論モードに切り替える
        self.policy_net.eval()

        # Q(s_t, a_t)を求める
        # self.policy_net(state_batch)は、[torch.FloatTensor of size self.batch_sizex2]になっており、
        # 実行したアクションに対応する[torch.FloatTensor of size self.batch_sizex1]にするために
        # gatherを使用します。
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # max{Q(s_t+1, a)}値を求める。
        # 次の状態がない場合は0にしておく
        next_state_values = Variable(torch.zeros(
            self.batch_size).type(torch.FloatTensor))
        next_state_values = next_state_values.to(self.device)

        # DDQN: use current network to evaluate action
        a_m = Variable(torch.zeros(self.batch_size).type(torch.LongTensor)).to(self.device)
        a_m[non_final_mask] = self.policy_net(non_final_next_states).max(1)[1].detach()
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態がある場合の値を求める
        # 出力であるdataにアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてその値（index=0）を出力します
        #next_state_values[non_final_mask] = self.target_net(non_final_next_states).data.max(1)[0].detach()
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 教師となるQ(s_t, a_t)値を求める
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        # ネットワークを訓練モードに切り替える
        self.policy_net.train()  # TODO: No need?

        # 損失関数を計算する。smooth_l1_lossはHuberlossです
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)

        # ネットワークを更新します
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

        # Update priority
        if self.prioritized and indexes != None:
            for i, val in enumerate(state_action_values):
                td_err = abs(expected_state_action_values[i].item() - val.item())
                self.memory.update(indexes[i], td_err)

    def _infer_action(self, state):
        self.policy_net.eval()  # ネットワークを推論モードに切り替える

        # Set device type; GPU or CPU
        input = Variable(state)
        input = input.to(self.device)

        # Infer
        return self.policy_net(input)

    def decide_action(self, state, episode):
        if self.boltzmann:
            # Boltzmann selection
            output = self._infer_action(state)
            prob = F.softmax(output, dim=1)
            action = torch.multinomial(prob, 1)
        else:
            # ε-greedy法で徐々に最適行動のみを採用する
            epsilon = 0.5 * (1 / (episode + 1))

            if epsilon <= np.random.uniform(0, 1):
                output = self._infer_action(state)
                action = output.data.max(1)[1].view(1, 1)
            else:
                # Generate random value [0.0, 1.0)
                action = torch.LongTensor([[random.randrange(self.num_actions)]])
                action = action.to(self.device)

        return action  # FloatTensor size 1x1

    def save_model(self, path):
        # Save a model checkpoint.
        print('Saving model...: {}'.format(path))
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        print('Loading model...: {}'.format(path))
        model = torch.load(path)
        self.policy_net.load_state_dict(model)
        self.update_target_network()

    def save_memory(self, path):
        print('Saving memory (size={})...: {}'.format(len(self.memory) ,path))
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, path):
        print('Loading memory...: {}'.format(path))
        with open(path, 'rb') as f:
            self.memory =  pickle.load(f)
            print('Loaded memory (size={})'.format(len(self.memory)))

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())