import logging
import os.path as osp
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import Logger

from configs.config_global import DEVICE, MAP_LOC, NP_SEED, TCH_SEED, USE_CUDA

class ThresholdFunc():
    def __init__(self, threshold=0.001):
        self.threshold = threshold

    def predict(self, x):
        x_diff = x[:, :int(x.shape[-1]/2)]- x[:, int(x.shape[-1]/2):]
        y_pred = np.sum(x_diff, axis=1) > self.threshold

        return y_pred.astype(int)

class Decoder(nn.Module):
    def __init__(self, input_dim, Params, output_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.n_hidden_units = Params['n_hidden_units']
        self.n_hidden_layers = Params['n_hidden_layers']
        self.output_dim = output_dim

        net = nn.Sequential()
        if self.n_hidden_layers == 0:
            net.add_module('fc_%d'%0, nn.Linear(self.input_dim, self.output_dim))
        else:
            net.add_module('fc_%d'%0, nn.Linear(self.input_dim, self.n_hidden_units))
            net.add_module('act_%d'%0, nn.ReLU())
            for i in range(1, self.n_hidden_layers):
                net.add_module('fc_%d'%i, nn.Linear(self.n_hidden_units, self.n_hidden_units))
                net.add_module('act_%d'%i, nn.ReLU())
            net.add_module('out_layer', nn.Linear(self.n_hidden_units, self.output_dim))

        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x

    def predict(self, x):
        if type(x) is np.ndarray:
            x = torch.Tensor(x).to(DEVICE)
        otp = self.forward(x)
        _, y_pred = torch.max(otp.detach(), 1)
        return y_pred.detach().cpu().numpy()

def sample_batch(data, labels, batch_size):
    sample_idx = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
    batch = torch.Tensor(data[sample_idx])
    batch_label = torch.Tensor(labels[sample_idx])

    return batch.to(DEVICE), batch_label.to(DEVICE, dtype=torch.long)

def train_decoder(
    config, 
    data, 
    labels, 
    Params, 
    batch_size=128, 
    iter_num=int(5e+3), 
    logger_name='decoder.txt', 
    log_every=None,
    decoder=None,
    logger=None
):
    
    if logger is None:
        logger = Logger(config.save_path, logger_name)

    # np.random.seed(NP_SEED + config.seed)
    # torch.manual_seed(TCH_SEED + config.seed)
    
    if USE_CUDA:
        logging.info("training with GPU")

    if log_every is None:
        log_every = iter_num // 10

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=1)

    if decoder is None:
        decoder = Decoder(data.shape[-1], Params).to(DEVICE)

    optimizer = optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    criteria = nn.CrossEntropyLoss()

    print('Training decoder...')
    i_log = 0
    testloss_list = []
    train_loss = 0.0
    best_acc = 0

    for i_b in range(iter_num):

        loss = 0.0
        optimizer.zero_grad()
        inp_, label_ = sample_batch(train_data, train_labels, batch_size)
        
        output = decoder(inp_)
        loss = criteria(output, label_)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

        if i_b % log_every == log_every - 1:

            correct = 0
            total = 0
            test_loss = 0.0
            test_b = 0

            with torch.no_grad():

                n = len(test_data)
                for idx in range((n - 1) // batch_size + 1):
                    inp_ = torch.from_numpy(test_data[batch_size * idx: batch_size * (idx + 1)]).to(DEVICE)
                    label_ = torch.from_numpy(test_labels[batch_size * idx: batch_size * (idx + 1)]).to(DEVICE, dtype=torch.long)

                    otp = decoder(inp_)
                    loss = criteria(otp, label_)
                    test_loss += loss
                    _, pred_id = torch.max(otp.detach(), 1)
                    total += label_.size(0)
                    correct += (pred_id == label_).sum().item()
                    test_b += 1

                test_acc = 100 * correct / total
                avg_testloss = test_loss / test_b

                best_acc = max(best_acc, test_acc)

            logger.log_tabular('BatchNum', i_b)
            logger.log_tabular('DataNum', i_b * batch_size)
            logger.log_tabular('TrainLoss', train_loss / log_every)
            logger.log_tabular('TestLoss', avg_testloss.item()) 
            logger.log_tabular('TestAcc', test_acc)
            logger.dump_tabular()

            train_loss = 0.0
            i_log += 1
            testloss_list.append(avg_testloss)

    # decoder.load_state_dict(torch.load(osp.join(config.save_path, 'decoder_best.pth'), map_location=MAP_LOC), strict=True)

    return decoder, logger, best_acc

