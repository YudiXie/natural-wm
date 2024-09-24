import os.path as osp
import numpy as np
import torch

from configs.configs import BaseConfig, ContinuousReportConfig
from models.model import AttCNNtoRNN
from utils.train_utils import model_init, task_init
from datasets.dataloader import TrialDataset
from tasks.continuousreport import ContinuousReport
from utils.train_utils import env_dict
from configs.config_global import DEVICE
from neurogym.utils.spaces import Box, Discrete
from tasks.tasktools import add_noise, random

def continuous_report_nontarget_distribution(config: ContinuousReportConfig, num_iters=200):

    # disable minimal pairwise distance
    config.minimal_pairwise_distance = 0

    # turn the config to keyword arguments
    cfg_kwargs = {}
    for name in dir(config):
        if name[0] != '_':
            cfg_kwargs[name] = getattr(config, name)

    # initialize dataset
    env = env_dict[config.task_type](**cfg_kwargs)

    def test_step(self: ContinuousReport, labels, outputs, trial_info, **unused):
        output = torch.stack(outputs)

        for i, trial in enumerate(trial_info):
            pred = output[trial['trial_length'] - 1, i]
            pred_ang = torch.atan2(pred[1], pred[0]).item()
            set_size = trial['num_patches']

            for idx, color in enumerate(trial['color_angles']):

                # only consider non-target colors
                if idx == trial['cue_idx']:
                    continue

                delta = (pred_ang - color) / np.pi * 180
                delta -= (delta > 180) * 360
                delta += (delta < -180) * 360
                self.errors[set_size].append(delta)

    def after_test_callback(self: ContinuousReport, save_path, **unused):
        for set_size in self.num_patches:
            error = self.errors[set_size]
            torch.save(error, osp.join(save_path, f'nontarget_error_{set_size}.pth'))

    dataset = TrialDataset(env, batch_size=config.batch_size, num_workers=0)
    image_size = (dataset.env.datum_size[2], dataset.env.datum_size[0], dataset.env.datum_size[1])
    
    out_space = env.action_space
    if isinstance(out_space, Box):
        out_size = out_space.shape[0]
    else:
        out_size = out_space.n

    config.model_class_size = out_size
    net = AttCNNtoRNN(config, image_size).to(DEVICE)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), map_location=DEVICE))
    dataset.env.before_test_callback()

    with torch.no_grad():

        for i in range(num_iters):

            inputs, labels, loss_mask, trial_info = dataset()
            inputs, labels, loss_mask = inputs.to(DEVICE), labels.to(DEVICE), loss_mask.to(DEVICE)

            output_list = []
            hidden = net.init_hidden(config.batch_size)

            if isinstance(config.input_noise, (list, tuple)):
                noise_std = random.choice(config.input_noise)
            else:
                noise_std = config.input_noise

            for i_step in range(inputs.shape[0]):

                # convert inputs to tensor
                step_input = inputs[i_step]
                step_input_w_noise = add_noise(step_input, noise_std)

                output, hidden = net(step_input_w_noise, hidden)
                output_list.append(output)

            test_step(
                dataset.env,
                model=net,
                inputs=inputs, 
                labels=labels, 
                loss_mask=loss_mask, 
                trial_info=trial_info,
                outputs=output_list
            )

    after_test_callback(dataset.env, save_path=config.save_path)