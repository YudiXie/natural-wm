import os
import os.path as osp
from datetime import datetime
import logging
import copy

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import models.model
import models.sensory_model
from tasks.continuousreport import ContinuousReport, CuedContinuousReport
from tasks.continuousreportseq import ContinuousReportSeq
import utils.model_utils as model_utils
from configs.config_global import DEVICE, MAP_LOC, ROOT_DIR
from tasks import taskfunctions
from tasks.changedetection import ChangeDetection
from tasks.dms import DelayedMatch


env_dict = {
    'delayed_match': DelayedMatch,
    'change_detection': ChangeDetection,
    'continuous_report': ContinuousReport,
    'cued_continuous_report': CuedContinuousReport,
    'sequential_continuous_report': ContinuousReportSeq
}


def config2dict(config):
    """
    convert a config object to a dictionary
    """
    config_dict = copy.deepcopy(config.__dict__)
    return config_dict


def log_complete(exp_path: str, start_time=None):
    """
    create a file to indicate the training is finished
    """
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    complete_time = datetime.now()
    with open(osp.join(exp_path, 'train_complete.txt'), 'w') as f:
        f.write(f'Training is complete at: {complete_time.strftime("%Y-%m-%d %H:%M:%S")}')
        if start_time is not None:
            f.write(f'\nTraining time: {str(complete_time - start_time)}')
    
    logging.info(f'Training is complete at: {complete_time.strftime("%Y-%m-%d %H:%M:%S")}')


def get_grad_norm(model: nn.Module):
    g = 0
    for param in model.parameters():
        g += param.grad.square().sum()
    return g


def grad_clipping(model, max_norm, printing=False):
    p_req_grad = [p for p in model.parameters() if p.requires_grad]

    if printing:
        grad_before = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_before += param_norm.item() ** 2
        grad_before = grad_before ** (1. / 2)

    clip_grad_norm_(p_req_grad, max_norm)

    if printing:
        grad_after = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_after += param_norm.item() ** 2
        grad_after = grad_after ** (1. / 2)

        if grad_before > grad_after:
            print("clipped")
            print("before: ", grad_before)
            print("after: ", grad_after)


def model_init(config_, datum_sizes, mode, net_num=None):
    """
    datum_size: tuple when not joint training, list of tuples when joint training
    """
    if net_num is not None:
        assert mode == 'eval', 'net number only provided at eval mode'

    if not config_.joint_train:
        datum_size = datum_sizes[0]
        assert type(datum_size) is tuple, 'datum size must be tuple'
    else:
        datum_size = datum_sizes
        assert type(datum_size) is list, 'datum size must be list of tuples'
        assert all([type(d_s) is tuple for d_s in datum_size]), 'datum size must be list of tuples'

    # initialize network
    if config_.model_type == 'LeNet':
        model = models.sensory_model.LeNet(datum_size)
    elif config_.model_type == 'ResNet':
        model = models.sensory_model.ResNet(
            config_.resblock_config, datum_size, 
            norm_layer=config_.cnn_norm,
            width=config_.cnn_width,
            num_classes=config_.model_class_size
        )
    elif config_.model_type == 'M5':
        model = models.sensory_model.M5(n_output=config_.model_class_size)
    elif config_.model_type == 'ResNetSimCLR':
        model = models.sensory_model.ResNetSimCLR(config_, datum_size)
    elif config_.model_type == 'AttCNNtoRNN':
        model = models.model.AttCNNtoRNN(config_, datum_size)
    else:
        raise NotImplementedError('model not implemented')
    model = model.to(DEVICE)

    if config_.load_path is not None:
        # if has designated path
        state_dict = torch.load(config_.load_path, map_location=MAP_LOC)
        model.load_state_dict(state_dict, strict=True)

        if hasattr(model, 'cnn') and config_.freeze_cnn:
            model.cnn.requires_grad_(False)
            model.cnn.train(False)

    elif mode == 'train':
        # model.train()
        pass

    elif mode == 'eval':
        # this part is broken
        model_state_dict, model_path = model_utils.get_model_state_dict(config_, net_num)
        rnn_state_dict = model_utils.get_rnn_para(model_state_dict)
        model.rnn.load_state_dict(rnn_state_dict, strict=True)
        out_state_dict = model_utils.get_out_layer_para(model_state_dict)
        model.out_layer.load_state_dict(out_state_dict, strict=True)
        model_path_prnt = 'CNN: ' + cnn_path + '\nRNN: ' + model_path

        print("successfully loaded model:\n" + model_path_prnt)
        model.eval()
    else:
        raise NotImplementedError('wrong model init mode')

    return model


def task_init(config_):
    """initialize tasks"""
    task_type = config_.task_type

    if task_type == "classification":
        task_func_ = taskfunctions.Classification(config_)
    elif task_type == 'contrastive_learning':
        task_func_ = taskfunctions.ContrastiveLearning(config_)
    else:
        raise NotImplementedError('task not implemented')

    return task_func_
