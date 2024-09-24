import logging
import os.path as osp
from datetime import datetime

import numpy as np
import torch
import random as sysrandom

from configs.config_global import (DEVICE, LOG_LEVEL, NP_SEED, TCH_SEED,
                                   USE_CUDA, DATA_DIR)
from configs.configs import BaseConfig
from datasets.dataloader import TrialDataset
from models.model import AttCNNtoRNN
from utils.config_utils import load_config
from utils.train_utils import env_dict
from neurogym.utils.spaces import Box, Discrete
from tasks.tasktools import ImageTrialEnv
from train import model_test

def evaluate_from_path(path):
    """Train from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_eval(config)

def model_eval(config: BaseConfig, env_wrapper: ImageTrialEnv=None, set_seed=True, load_path=None):
    """
    Evaluate a model on a given task.
    :param config: config object. 
    :param env_wrapper: an optional environment wrapper for an ImageTrialEnv instance, can be used to modify callback 
        functions for added functionality. Please refer to the ImageTrialEnv class for details.
    :param set_seed: whether to set random seeds
    :param load_path: path to load the model from. Model will be loaded from config.save_path/net_best.pth if load_path is None.
    """
    if set_seed:
        np.random.seed(NP_SEED + config.seed)
        torch.manual_seed(TCH_SEED + config.seed)
        sysrandom.seed(config.seed)
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))
    start_time = datetime.now()

    # turn the config to keyword arguments
    cfg_kwargs = {}
    for name in dir(config):
        if name[0] != '_':
            cfg_kwargs[name] = getattr(config, name)

    # initialize dataset
    env = env_dict[config.task_type](**cfg_kwargs)
    if env_wrapper is not None:
        env = env_wrapper(env)

    dataset_kwargs = dict(
        batch_size=config.batch_size, num_workers=config.num_workers, 
        noise_std=config.input_noise, noise_mode=config.input_noise_mode, noise_res=config.input_noise_resolution
    )
    dataset = TrialDataset(env, **dataset_kwargs)

    # initialize network
    image_size = (dataset.env.datum_size[2], dataset.env.datum_size[0], dataset.env.datum_size[1])
    
    out_space = env.action_space
    if isinstance(out_space, Box):
        out_size = out_space.shape[0]
    else:
        out_size = out_space.n

    config.model_class_size = out_size
    net = AttCNNtoRNN(config, image_size).to(DEVICE)
    if load_path is None:
        load_path = osp.join(config.save_path, 'net_best.pth')
    net.load_state_dict(torch.load(load_path), strict=cfg_kwargs.get('strict_loading', True))

    dataset.env.before_test_callback(
        model=net,save_path=config.save_path,
        batch_num=config.max_batch
    )
    model_test(net, config, dataset)
    callback_results = dataset.env.after_test_callback(
        model=net,save_path=config.save_path,
        is_best=True, batch_num=config.max_batch
    )
    
    print('Finished testing. Time elapsed: {}'.format(datetime.now() - start_time))
    return callback_results