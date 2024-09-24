import os.path as osp
import random
from typing import List

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from scipy.stats import circstd

from configs.config_global import DEVICE
from configs.configs import (BaseConfig, ContinuousReportConfig, DMSConfig,
                             LuckVogelConfig)
from datasets.data_sets import get_class_size
from tasks.tasktools import add_noise
from utils.logger import Logger

def data_batch_to_device(data_b, device=DEVICE):
    if type(data_b) is torch.Tensor:
        return data_b.to(device)
    elif type(data_b) is tuple or type(data_b) is list:
        return [data_batch_to_device(data, device) for data in data_b]
    else:
        raise NotImplementedError("input type not recognized")

class TaskFunction:

    def __init__(self, config: BaseConfig):
        """
        initailize information from config
        """
        pass

    def roll(self, model: nn.Module, data_batch: tuple, train=False, test=False, evaluate=False):
        """
        In the training mode (train=True), this method should return a scalar representing the training loss
        In the eval or test mode, this method should return a tuple of at least 3 elements: 
        (
            test loss, 
            number of predictions (often equal to batchsize), 
            number of correct prediction (in classification tasks) or sum of error (in regression tasks),
            ... (additional information that might be used in callback functions)
        )
        """
        raise NotImplementedError

    def after_testing_callback(self, batch_info: List[tuple], logger: Logger, save_path: str, is_best: bool, batch_num: int):
        """
        An optional function called after the test stage
        :param batch_info: list of return values of task.roll(mode, data_batch, test=True) for batches of data in the test set
        :param is_best: whether the test result is the best one
        :param batch_num: number of batches used for training
        """
        pass

    def after_training_callback(self, config, model):
        """
        An optional function called after training is done
        :param model: the best model
        """
        pass


class Classification(TaskFunction):

    def __init__(self, config):
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = config.batch_size
        self.task_batch_s = self.batch_size

    def roll(self, model, data_batch, train=False, test=False, evaluate=False):
        assert train + test + evaluate == 1, "only one mode should be activated"
        input_, label_ = data_batch_to_device(data_batch)
        output = model(input_)

        task_loss = self.criterion(output, label_)

        pred_num = 0
        pred_correct = 0

        if test or evaluate:
            _, pred_id = torch.max(output.detach(), 1)
            pred_num += label_.size(0)
            pred_correct += (pred_id == label_).sum().item()

        if train:
            return task_loss
        elif test or evaluate:
            return task_loss, pred_num, pred_correct
        else:
            raise NotImplementedError("Not Implemented")


class ContrastiveLearning(TaskFunction):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.task_batch_s = self.batch_size
        self.temperature = 0.07
        self.criterion = torch.nn.CrossEntropyLoss()

    def info_nce_loss(self, features):
        # adapted from https://github.com/sthalles/SimCLR
        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)],
                           dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(DEVICE)

        features = func.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (2 * self.batch_size,
                                           2 * self.batch_size)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

        logits = logits / self.temperature
        return logits, labels

    def roll(self, model, data_batch, train=False, test=False, evaluate=False):
        assert train + test + evaluate == 1, "only one mode should be activated"
        input_, label_ = data_batch_to_device(data_batch)

        input_ = torch.cat(input_, dim=0)
        features = model(input_)
        logits, labels = self.info_nce_loss(features)
        task_loss = self.criterion(logits, labels)

        if train:
            return task_loss
        else:
            raise NotImplementedError("Not Implemented")