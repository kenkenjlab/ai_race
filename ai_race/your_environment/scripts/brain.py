#!python3
#-*- coding: utf-8 -*-

import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from replaymemory import ReplayMemory, Transition
import torchvision.models as models
#from samplenet import SampleNet

#------------------------------------------------

class Brain:
    TARGET_UPDATE = 10
    def __init__(self, num_actions, batch_size = 32, capacity = 10000, gamma = 0.99):
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_actions = num_actions

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(capacity)

        # Build network
        self.policy_net = models.resnet18()
        self.policy_net.fc = torch.nn.Linear(512, self.num_actions)
        self.target_net = models.resnet18()
        self.target_net.fc = torch.nn.Linear(512, self.num_actions)
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
        transitions = self.memory.sample(self.batch_size)

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

        # 次の状態がある場合の値を求める
        # 出力であるdataにアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてその値（index=0）を出力します
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).data.max(1)[0].detach()

        # 教師となるQ(s_t, a_t)値を求める
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # ネットワークを訓練モードに切り替える
        self.policy_net.train()  # TODO: No need?

        # 損失関数を計算する。smooth_l1_lossはHuberlossです
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # ネットワークを更新します
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

    def decide_action(self, state, episode):
        # ε-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.policy_net.eval()  # ネットワークを推論モードに切り替える

            # Set device type; GPU or CPU
            input = Variable(state)
            input = input.to(self.device)

            # Infer
            output = self.policy_net(input)
            action = output.data.max(1)[1].view(1, 1)

        else:
            # Generate random value [0.0, 1.0)
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
            action = action.to(self.device)

        return action  # FloatTensor size 1x1

    def save(self, path):
        # Save a model checkpoint.
        print('Saving model...: {}'.format(path))
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        model = torch.load(path)
        self.policy_net.load_state_dict(model)
        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())