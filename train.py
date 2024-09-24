import logging
import os.path as osp
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import random as sysrandom
import torchvision

from configs.config_global import (DEVICE, LOG_LEVEL, NP_SEED, TCH_SEED,
                                   USE_CUDA, DATA_DIR)
from configs.configs import BaseConfig
from datasets.dataloader import TrialDataset
from models.model import AttCNNtoRNN
from utils.config_utils import load_config
from utils.train_utils import env_dict, grad_clipping, log_complete, config2dict
from utils.logger import Logger
from neurogym.utils.spaces import Box, Discrete
from tasks.tasktools import add_noise, random, get_noise

import wandb


def train_from_path(path):
    """Train from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_train(config)


def model_test(
    net: nn.Module, 
    config: BaseConfig, 
    dataset: TrialDataset,
):
    with torch.no_grad():

        test_loss = 0
        pred_num = 0
        pred_correct = 0

        for i in range(config.test_batch):

            inputs, labels, loss_mask, trial_info = dataset()
            inputs, labels, loss_mask = inputs.to(DEVICE), labels.to(DEVICE), loss_mask.to(DEVICE)

            hidden_list = []
            output_list = []
            hidden = net.init_hidden(config.batch_size)
            loss = 0
            
            for i_step in range(inputs.shape[0]):

                # convert inputs to tensor
                step_input = inputs[i_step]
                if config.input_resolution is not None:
                    step_input = torchvision.transforms.functional.resize(step_input, size=config.input_resolution)
                # convert targets to tensor
                step_target = labels[i_step]
                if config.print_mode == 'accuracy':
                    step_target = step_target.long()
                mask = loss_mask[i_step]

                output, hidden = net(step_input, hidden)
                hidden_list.append(hidden)
                output_list.append(output)
                loss += (dataset.env.criterion(output, step_target) * mask).sum().item()

                if config.print_mode == 'accuracy':
                    pred_correct += ((output.argmax(-1) == step_target) * mask).sum().item()

            output_list = torch.stack(output_list, dim=0)
            hidden_list = torch.stack(hidden_list, dim=0)
            dataset.env.test_step(
                model=net,
                inputs=inputs, 
                labels=labels, 
                loss_mask=loss_mask, 
                trial_info=trial_info,
                hiddens=hidden_list,
                outputs=output_list
            )

            pred_num += loss_mask.sum().item()
            loss = loss / loss_mask.sum().item()
            test_loss += loss

        test_loss /= config.test_batch
        
        test_results = {}
        if config.print_mode == 'accuracy':
            test_results['TestAcc'] = 100 * pred_correct / pred_num
        test_results['TestLoss'] = test_loss
    return test_results


def model_train(config: BaseConfig):
    """
        train a model based on config, return the trained model while logging performance
    """
    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    sysrandom.seed(config.seed)
    # set the torch hub directory
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))
    start_time = datetime.now()

    assert config.config_mode == 'train', 'config mode must be train'

    wandb.init(project="multi-system-wm", config=config2dict(config))
    logger = Logger(output_dir=config.save_path, exp_name=config.experiment_name)

    # gradient clipping
    if config.grad_clip is not None:
        logging.info(f"Performs grad clipping with max norm {config.grad_clip}")

    # turn the config to keyword arguments
    cfg_kwargs = {}
    for name in dir(config):
        if name[0] != '_':
            cfg_kwargs[name] = getattr(config, name)

    # initialize dataset
    # dataset = ngym.Dataset(env, batch_size=config.batch_size, seq_len=config.seq_len)
    env = env_dict[config.task_type](**cfg_kwargs)
    dataset_kwargs = dict(
        batch_size=config.batch_size, num_workers=config.num_workers, 
        noise_std=config.input_noise, noise_mode=config.input_noise_mode, noise_res=config.input_noise_resolution
    )
    dataset = TrialDataset(env, **dataset_kwargs)
    if config.perform_test:
        test_dataset = TrialDataset(env, **dataset_kwargs)
    
    criterion = dataset.env.criterion

    # initialize network
    # PIL Image or numpy.ndarray has dimension (H x W x C) in the range [0, 255]
    # torch.FloatTensor has the dimension (C x H x W) in the range [0.0, 1.0]
    # in the environment we specify in the PIL and numpy convention
    image_size = (dataset.env.datum_size[2], dataset.env.datum_size[0], dataset.env.datum_size[1])
    
    out_space = env.action_space
    if isinstance(out_space, Box):
        out_size = out_space.shape[0]
    else:
        out_size = out_space.n

    config.model_class_size = out_size
    net = AttCNNtoRNN(config, image_size).to(DEVICE)
    
    if config.load_path is not None:
        net.load_state_dict(torch.load(config.load_path))

    # initialize optimizer
    if config.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.wdecay)
    elif config.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.wdecay, amsgrad=True)
    elif config.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr_SGD,
                                    momentum=0.9, weight_decay=config.wdecay)
    else:
        raise NotImplementedError('optimizer not implemented')

    # initialize Learning rate scheduler
    if config.use_lr_scheduler:

        # By default, run sched.step() after performance plateaus
        if config.scheduler_type == 'ExponentialLR':
            scheduler = lrs.ExponentialLR(optimizer, gamma=0.3)
            scheduler_steps = 0
        elif config.scheduler_type == 'StepLR':
            scheduler = lrs.StepLR(optimizer, 1, gamma=0.3)
            scheduler_steps = 0
        else:
            raise NotImplementedError('scheduler_type must be specified')

    i_log = 0
    loss_list = []
    train_loss = 0.0

    pred_correct = 0
    pred_num = 0

    for i_batch in range(config.max_batch):

        start_time = datetime.now()

        # re-initialize dataset
        if str(i_batch) in config.curriculum:

            cfg_kwargs.update(config.curriculum[str(i_batch)])
            env = env_dict[cfg_kwargs['task_type']](**cfg_kwargs)
            dataset = TrialDataset(env, **dataset_kwargs)
            criterion = dataset.env.criterion
            net.update_config(**cfg_kwargs)

        inputs, labels, loss_mask, trial_info = dataset()
        inputs, labels, loss_mask = inputs.to(DEVICE), labels.to(DEVICE), loss_mask.to(DEVICE)

        # save model
        if (i_batch + 1) % config.save_every == 0:
            torch.save(net.state_dict(), osp.join(config.save_path, 'net_{}.pth'.format(i_batch + 1)))

        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=DEVICE)
        hidden = net.init_hidden(config.batch_size)

        if isinstance(config.input_noise, (list, tuple)):
            noise_std = random.choice(config.input_noise)
        else:
            noise_std = config.input_noise
        
        for i_step in range(inputs.shape[0]):

            # convert inputs to tensor
            step_input = inputs[i_step]
            if config.input_resolution is not None:
                step_input = torchvision.transforms.functional.resize(step_input, size=config.input_resolution)

            # convert targets to tensor
            step_target = labels[i_step]
            if config.print_mode == 'accuracy':
                step_target = step_target.long()
            mask = loss_mask[i_step]

            output, hidden = net(step_input, hidden)

            loss += (criterion(output, step_target) * mask).sum()
            if config.print_mode == 'accuracy':
                pred_correct += ((output.argmax(-1) == step_target) * mask).sum().item()
        
        pred_num += loss_mask.sum().item()
        loss = loss / loss_mask.sum().item()
        loss.backward()

        # gradient clipping
        if config.grad_clip is not None:
            grad_clipping(net, config.grad_clip)

        optimizer.step()
        train_loss += loss.item()

        metrics = {"train/BatchNum": i_batch + 1,
                   "train/DataNum": (i_batch + 1) * config.batch_size,
                   "train/TrainLoss": loss.item()}
        if config.print_mode == 'accuracy':
            metrics["train/TrainAcc"] = 100 * pred_correct / pred_num

        # log performance
        if i_batch % config.log_every != config.log_every - 1:
            wandb.log(metrics)
        else:
            avg_train_loss = train_loss / config.log_every

            pred_correct = 0
            pred_num = 0
            train_loss = 0.0
            i_log += 1

            # Begin of test
            test_dataset.env.before_test_callback(
                model=net,save_path=config.save_path,
                batch_num=i_batch
            )
            if config.perform_test:
                test_results = model_test(net, config, test_dataset)
                test_loss = test_results['TestLoss']
                metrics.update({f'test/{k}': v for k, v in test_results.items()})
            else:
                test_loss = avg_train_loss
            loss_list.append(test_loss)

            # save the model with best testing loss
            if test_loss <= min(loss_list):
                torch.save(net.state_dict(),
                           osp.join(config.save_path, 'net_best.pth'))
                best = True
            else:
                best = False
            
            callback_results = test_dataset.env.after_test_callback(
                model=net,save_path=config.save_path,
                is_best=best, batch_num=i_batch
            )
            if callback_results is not None:
                metrics.update({f'test/{k}': v for k, v in callback_results.items()})
            
            # End of test, log metrics, also print the metrics to the console
            wandb.log(metrics)
            logger.dump_tabular_dict(metrics)

            # check plateau
            if config.early_stop \
                    and (i_log >= 2 * config.eslen) \
                    and (min(loss_list[-config.eslen:]) > min(loss_list[:-config.eslen])):
                if config.use_lr_scheduler:
                    scheduler.step()
                    logging.info("scheduler step")
                    i_log = 0
                    scheduler_steps += 1
                    if scheduler_steps >= 4:
                        break
                else:
                    break
    
    log_complete(config.save_path, start_time)

    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth')))
    dataset.env.after_training_callback(config=config, model=net)

    wandb.summary['minimum_loss'] = min(loss_list)
    wandb.alert(
            title='Run Finished',
            text=f'Run Finished, Minimum Loss: {min(loss_list):.3f}'
        )
    wandb.finish()

    return net