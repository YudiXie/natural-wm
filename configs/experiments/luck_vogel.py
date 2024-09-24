__all__ = [
    'luckvogel_ngym',
    'luckvogel_change_magnitude', 
    'luckvogel_change_magnitude_test',
    'luckvogel_pret',
    'luckvogel_model_size',
    'luckvogel_diff_cnn',
    'luckvogel_compare_att',
    'luckvogel_noatt',
]

from collections import OrderedDict
from copy import deepcopy
from configs.configs import LuckVogelConfig
from utils.config_utils import config_dict2config_df, vary_config
import numpy as np
import os.path as osp

def luckvogel_ngym():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_ngym'
    
    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['none', 'Classification_CIFAR10']
    config_ranges['freeze_cnn'] = [False, True]

    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)
    return configs

def luckvogel_change_magnitude():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_change_magnitude'
    config.change_magnitude = None
    config.use_fixed_colors = False

    config_ranges = OrderedDict()
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def luckvogel_change_magnitude_test():
    configs = luckvogel_change_magnitude()
    new_configs = {0: [], 1: [], 2: [], 3: []}

    for i in range(len(configs)):
        for angle in range(0, 100, 10):

            cfg: LuckVogelConfig = deepcopy(configs.loc[i].config)

            cfg.load_path = osp.join(cfg.save_path, 'net_best.pth')
            cfg.lr = 0
            cfg.max_batch = cfg.log_every = 1
            cfg.num_workers = cfg.cpu = 1
            cfg.change_magnitude = angle / 180 * np.pi
            cfg.curriculum = {}

            original_exp_name = cfg.experiment_name
            cfg.experiment_name = 'luckvogel_change_magnitude_test'
            cfg.save_path = cfg.save_path.replace(original_exp_name, cfg.experiment_name)
            cfg.save_path += f'_angle{angle}'

            new_configs[cfg.seed].append(cfg)

    return config_dict2config_df(new_configs)

def luckvogel_pret():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_pret'

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = [
        'Classification_CIFAR10', 
        'Classification_LuckVogel', 
        'Contrastive_CIFAR10', 
        'Contrastive_LuckVogel'
    ]
    config_ranges['freeze_cnn'] = [False, True]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def luckvogel_noatt():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_noatt'
    config.att = 'none'

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['none', 'Classification_CIFAR10']
    config_ranges['freeze_cnn'] = [False, True]

    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)
    return configs

def luckvogel_model_size():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_model_size'
    config.neurogym = False
    config.gpu_constraint = 'ampere'

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['none',
                                 'none', 
                                 'none', 
                                 'none', 
                                 'none', 
                                 'Classification_CIFAR10',
                                 'Classification_CIFAR10',
                                 'Classification_CIFAR10',
                                 'Classification_CIFAR10',
                                 'Classification_CIFAR10',
                                 'Classification_MNIST',
                                 'Contrastive_CIFAR10',
                                 'none']
    
    config_ranges['freeze_cnn'] = [False,
                                   False,
                                   False,
                                   False,
                                   False,
                                   True,
                                   True,
                                   True,
                                   True,
                                   True,
                                   True,
                                   True,
                                   True,]
    config_ranges['cnn_width'] = [64, 64, 64, 32, 16, 
                                  64, 64, 64, 32, 16,
                                  64, 64, 64]
    config_ranges['hidden_size'] = [128, 64, 32, 128, 128,
                                    128, 64, 32, 128, 128,
                                    128, 128, 128]
    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)
    return configs

def luckvogel_diff_cnn():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_diff_cnn'

    config.gpu_constraint = '18GB'
    config.att = 'none'
    config.input_resolution = (224, 224)
    config.input_noise_mode = 'per_period'

    config_ranges = OrderedDict()
    config_ranges['cnn_archi'] = ['ResNet-18', 'ResNet-50', 'ViT-B', 'ResNet-50', 'ResNet-50'] + \
                                ['AlexNet', 'ResNet-18', 'ResNet-50', 'ViT-B'] * 2
    config_ranges['cnn_pret'] = ['R3M', 'R3M', 'VC-1', 'autoencoding_taskonomy', 'inpainting_taskonomy'] + \
                                ['Classification_ImageNet'] * 4 + ['none'] * 4
    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)
    return configs

def luckvogel_compare_att():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_compare_att'
    config.neurogym = False

    config_ranges = OrderedDict()
    config_ranges['att'] = ['none', 'cbam', 'film', 'elementwise', ]

    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)
    return configs