import torch
import torch.nn as nn
import torchvision.transforms as T
import models.sensory_model as sensory_model
import numpy as np

from configs.config_global import DEVICE
from configs.configs import BaseConfig
from tasks.tasktools import random
from utils.model_utils import get_att, get_cnn, get_rnn

class AttCNNtoRNN(nn.Module):

    def __init__(self,
                 config: BaseConfig,
                 image_size):
        super().__init__()

        self.cnn_archi = cnn_archi = config.cnn_archi
        self.rnn_type = rnn_type = config.rnn
        self.att_type = att_type = config.att

        self.hidden_size = config.hidden_size
        self.cnn = get_cnn(config, image_size)

        if att_type != 'none':
            if cnn_archi == 'LeNet':
                # use hook to specify layers that would be applied attention
                self.cnn.pool2.register_forward_hook(self.cnn_feature_hook)

            elif cnn_archi in ['ResNet', 'MultiscaleResNet']:
                self.cnn.cnn: sensory_model.ResNet
                cnn_layers = [self.cnn.cnn.layer1[-1].conv2,
                            self.cnn.cnn.layer2[-1].conv2,
                            self.cnn.cnn.layer3[-1].conv2]
                # apply attention at the 2nd conv in each resblock, following the FiLM paper 
                for layer in config.att_layers:
                    cnn_layers[layer].register_forward_hook(self.cnn_feature_hook)
            else:
                raise NotImplementedError('Attention not implemented for this CNN architecture')

        self.hid_out = None
        self.cnn_features = []
        with torch.no_grad():
            sample_input = torch.randn(1, *image_size)
            if config.input_resolution is not None:
                sample_input = T.functional.resize(sample_input, config.input_resolution)
            cnn_out = self.cnn(sample_input)
        self.feature_shapes = [feature.shape[1:] for feature in self.cnn_features]

        self.att = []
        for feature_shape in self.feature_shapes:
            self.att.append(get_att(self.att_type, self.hidden_size, feature_shape))
        self.att = nn.ModuleList(self.att)
        self.rnn_in_size = cnn_out.view(-1).shape[0]

        # An optional projection layer to reduce the dimensionality of the cnn output
        # self.cnn_ro_rnn_proj = nn.Sequential(nn.Linear(self.rnn_in_size, self.hidden_size), nn.ReLU())
        # self.rnn_in_size = self.hidden_size

        self.rnn = get_rnn(
            rnn_type, self.rnn_in_size, self.hidden_size, 
            config.dt / config.rnn_eta, config.layernorm,
            config.additive_rnn_noise, config.rnn_noise
        )
        self.freeze_rnn = config.freeze_rnn
        self.out_layer = nn.Linear(self.hidden_size, config.model_class_size)
        self.use_pos_encoding = config.use_pos_encoding
        if self.use_pos_encoding:
            self.pos_encoding = PositionalEncoding(self.rnn_in_size, max_len=500)

    def init_hidden(self, batch_size):
        if self.freeze_rnn:
            self.rnn.requires_grad_(False)
            self.att.requires_grad_(False)
        
        if self.rnn_type == 'LSTM':
            init_hid = (torch.zeros(batch_size, self.hidden_size).to(DEVICE),
                        torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        elif self.rnn_type in ['plainRNN', 'GRU']:
            init_hid = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        elif self.rnn_type in ['VanillaRNN', 'CTRNN', 'RNNSTSP']:
            init_hid = self.rnn.init_hidden(batch_size)
        else:
            raise NotImplementedError('RNN not implemented')
        self.hid_out = init_hid
        self.last_cnn_embedding = None
        self.last_input = None
        self.time_step = 0
        return init_hid

    def cnn_feature_hook(self, module, inp, otp):
        """
        Apply attention to output of some cnn layer.
        """
        idx = len(self.cnn_features)
        self.cnn_features.append(otp)
        if self.hid_out is None:
            return
        att_in = self.hid_out[0] if self.rnn_type in ['LSTM', 'RNNSTSP'] else self.hid_out
        att_out = self.att[idx](otp, att_in)
        return att_out

    def forward(self, inp, hid_in):
        if hid_in is None:
            hid_in = self.init_hidden(inp.shape[0])

        x = None
        if self.att_type == 'none' and self.last_input is not None:
            # TODO: change this to checking each element of the batch
            if (self.last_input[: 4] - inp[: 4]).abs().sum() < 1e-6:
                x = self.last_cnn_embeddings
       
        if x is None:
            self.cnn_features = []
            x = self.cnn(inp).reshape(inp.shape[0], -1)

        if self.att_type == 'none':
            self.last_input = inp
            self.last_cnn_embeddings = x

        if self.use_pos_encoding:
            x = self.pos_encoding(x, self.time_step)
            self.time_step += 1

        self.hid_out = self.rnn(x, hid_in)
        x = self.hid_out[0] if self.rnn_type in ['LSTM', 'RNNSTSP'] else self.hid_out
        x = self.out_layer(x)

        return x, self.hid_out

    def update_config(self, freeze_cnn=True, **unused):
        if freeze_cnn:
            self.cnn.requires_grad_(False)
        else:
            self.cnn.requires_grad_(True)

    def set_mode(self, mod):
        pass

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(max_len * 2) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, step=0) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[step]
        return x