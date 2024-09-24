import os
import numpy as np
import torch
import tqdm

from utils.model_utils import get_cnn
from tasks.changedetection import ChangeDetection
from configs.config_global import ROOT_DIR, FIG_DIR
from analysis.plots import adjust_figure
from configs.configs import MultiModelConfig

import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['figure.facecolor'] = 'w'
plt.rcParams.update({'font.size': 18})


def append_tuple(name, activity_dict, output):
    for i_, otp in enumerate(output):
        new_name = name + '_' + str(i_ + 1)
        if isinstance(otp, tuple):
            append_tuple(new_name, activity_dict, otp)
        elif isinstance(otp, torch.Tensor):
            activity_dict[new_name].append(otp.detach().cpu().numpy())
        else:
            raise NotImplementedError('append type not implemented')


def append_activations(name, activity_dict):
    """
    Returns a hook function that can be registered with model layers
        to obtain and store the output history of hidden activations in activation_dict
    name: the name of module to record activities
    activity_dict: a collection.defaultdict with default factory function set to list
    """
    assert isinstance(activity_dict, defaultdict) \
           and activity_dict.default_factory == list, 'activity_dict must be default dict'

    def hook(module, inp, otp):
        if isinstance(otp, torch.Tensor):
            activity_dict[name].append(otp.detach().cpu().numpy())
        elif isinstance(otp, tuple):
            append_tuple(name, activity_dict, otp)
        else:
            raise NotImplementedError('append type not implemented')

    return hook


if __name__ == '__main__':

    config = MultiModelConfig()
    fig_path = os.path.join(FIG_DIR, 'overall_activation')
    os.makedirs(fig_path, exist_ok=True)

    for seed in range(4):
        config.seed = seed
        net = get_cnn(config, (3, 32, 32))

        layers = ['layer1', 'layer2', 'layer3']

        # register forward hook for all layers to record activity from
        batch_activity = defaultdict(list)
        all_activity = defaultdict(list)
        handles = defaultdict(list)
        batch_info = []

        for layer in layers:
            handles[layer] = net.__getattr__(layer).register_forward_hook(
                append_activations(layer, batch_activity))

        # record activities
        num_image = 1000
        num_patch_list = np.arange(1, 13)

        with torch.no_grad():
            for num_patch in num_patch_list:
                env = ChangeDetection(num_patches=(num_patch,))
                for i in tqdm.tqdm(range(num_image)):
                    trial = env.new_trial()
                    # sample image
                    image1 = torch.from_numpy(env.ob[0]) / 255
                    image1 = image1.permute(2, 0, 1).unsqueeze(0)
                    image1 = image1 + torch.randn_like(image1) * config.input_noise
                    embed1 = net(image1)

        batch_activity_summary = {}
        for k, v in batch_activity.items():
            batch_activity_summary[k] = [np.mean(act) for act in v]

        # plot mean layer activation against number of patches for each layer
        for layer in layers:
            layer_data = [batch_activity_summary[layer][i * num_image:(i + 1) * num_image] for i in
                        range(len(num_patch_list))]

            plt.figure()
            plt.boxplot(layer_data, notch=True)
            plt.title(f'Conv {layer} Mean activation')
            plt.xlabel("Set size")
            plt.ylabel("Mean network activation")
            # plt.ylim([0.4, 0.8])
            adjust_figure()
            plt.savefig(os.path.join(fig_path, f'{layer}_activation{seed}.pdf'), transparent=True)

        # plot mean layer activation against number of patches, all layers in one plot
        plt.figure()
        for layer in layers:
            layer_data = [batch_activity_summary[layer][i * num_image:(i + 1) * num_image] for i in
                        range(len(num_patch_list))]

            plt.boxplot(layer_data, notch=True)
            plt.title(f'Conv {layer} Mean activation')
            plt.xlabel("Set size")
            plt.ylabel("Mean network activation")

        adjust_figure()
        plt.savefig(os.path.join(fig_path, f'all_layers_activation{seed}.pdf'), transparent=True)