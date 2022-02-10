import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh, Linear, \
    Flatten

from pommerman.nn.PommerModel import PommerModel
from pommerman.nn.builder_util import get_act, _Stem, _PolicyHead, _ValueHead


class A2CNet(PommerModel):

    def __init__(
        self,
        n_labels=6,
        channels=45,
        nb_input_channels=18,
        board_height=11,
        board_width=11,
        bn_mom=0.9,
        act_type="relu"
    ):
        """
        :param n_labels: Number of labels the for the policy
        :param channels: Used for all convolution operations. (Except the last 2)
        :param channels_policy_head: Number of channels in the bottle neck for the policy head
        :param channels_value_head: Number of channels in the bottle neck for the value head
        :param num_res_blocks: Number of residual blocks to stack. In the paper they used 19 or 39 residual blocks
        :param value_fc_size: Fully Connected layer size. Used for the value output
        :param bn_mom: Batch normalization momentum
        :return: net description
        """

        print("INIT A2C")

        super(A2CNet, self).__init__(nb_input_channels=nb_input_channels, board_height=board_height,
                                     board_width=board_height, is_stateful=False, state_batch_dim=None)

        self.has_state_input = False

        self.nb_flatten = board_height * board_width * channels

        self.body = Sequential(Conv2d(in_channels=nb_input_channels, out_channels=channels,
                                      kernel_size=(3, 3), padding=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels),
                               get_act(act_type),
                               Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=(3, 3), padding=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels),
                               get_act(act_type),
                               Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=(3, 3), padding=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels),
                               get_act(act_type),
                               Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=(3, 3), padding=(1, 1), bias=False),
                               BatchNorm2d(momentum=bn_mom, num_features=channels),
                               get_act(act_type),
                               Flatten(),
                               Linear(in_features=self.nb_flatten, out_features=1024),
                               get_act(act_type),
                               Linear(in_features=1024, out_features=512),
                               get_act(act_type),
                               Linear(in_features=512, out_features=512),
                               get_act(act_type),
                               Linear(in_features=512, out_features=512),
                               get_act(act_type)
                               )

        self.lstm = torch.nn.LSTMCell(512, 64)
        self.linear_after_lstm = nn.Linear(64, 64)

        # create the two heads which will be used in the hybrid fwd pass
        self.policy_head = nn.Linear(64, n_labels)
        self.value_head = nn.Linear(512, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() and torch.version.hip else "cpu")
        # TODO prüfen
        self.rnn_hidden_size = 64
        self.obs_width = board_width
        self.gamma = 0.99  # Discount factor for rewards (default 0.99)
        self.entropy_coef = 0.01  # Entropy coefficient (0.01)
        self.lr = 0.001  # 3e-2
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, flat_input, hn, cn):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Value & Policy Output
        """
        # input shape processing
        x, state_bf = self.unflatten(flat_input)

        out = self.body(x)
        hn, cn = self.lstm(out, (hn, cn))
        rnn_out = self.linear_after_lstm(hn)

        value = self.value_head(out)
        policy = self.policy_head(rnn_out)

        return policy, value, hn, cn

    def init_rnn(self):
        device = self.device
        s = self.rnn_hidden_size
        ret_1 = torch.zeros(s).detach().numpy()
        ret_2 = torch.zeros(s).detach().numpy()
        return ret_1, ret_2

    def discount_rewards(self, _rewards):
        R = 0
        gamma = self.gamma
        rewards = []
        for r in _rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)

        return rewards

    def get_state_shape(self, batch_size: int):
        raise NotImplementedError
