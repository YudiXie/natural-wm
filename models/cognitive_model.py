import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs.config_global import DEVICE

# Vanilla RNNCell
class VanillaRNNCell(nn.Module):
    
    def __init__(
            self, 
            input_size, 
            hidden_size, 
            nonlinearity="tanh", 
            ct=False, 
            alpha=0.1, 
            layernorm=False,
            mul_rnn_noise=0,
            add_rnn_noise=0
        ):

        super(VanillaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.ct = ct
        self.alpha = alpha
        self.mul_rnn_noise = mul_rnn_noise
        self.add_rnn_noise = add_rnn_noise

        self.weight_ih = nn.Parameter(torch.zeros((input_size, hidden_size)))
        self.weight_hh = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        if layernorm:
            self.layernorm = nn.LayerNorm((self.hidden_size, ), elementwise_affine=True)
        else:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.layernorm = None
        self.reset_parameters()

        if self.nonlinearity == "tanh":
            self.act = torch.tanh
        elif self.nonlinearity == "relu":
            self.act = F.relu
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inp, hidden_in):
        if not self.ct:
            pre_act = torch.matmul(inp, self.weight_ih) + torch.matmul(hidden_in, self.weight_hh) + self.bias
            hidden_out = self.act(pre_act)
        else:
            pre_act = torch.matmul(inp, self.weight_ih) + torch.matmul(hidden_in, self.weight_hh)
            if self.add_rnn_noise > 0:
                pre_act = pre_act + torch.randn_like(pre_act) * np.sqrt(2 / self.alpha) * self.add_rnn_noise
            if self.layernorm is not None:
                pre_act = self.layernorm(pre_act)
            else:
                pre_act = pre_act + self.bias
            hidden_out = (1 - self.alpha) * hidden_in + self.alpha * self.act(pre_act)
            if self.mul_rnn_noise > 0:
                hidden_out = hidden_out + torch.randn_like(hidden_out) * self.mul_rnn_noise
        return hidden_out

    def init_hidden(self, batch_s):
        return torch.zeros(batch_s, self.hidden_size).to(DEVICE)