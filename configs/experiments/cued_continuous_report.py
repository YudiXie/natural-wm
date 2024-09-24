__all__ = [
    'cued_continuous_report', 
    'cued_continuous_report_set_size',
    'cued_continuous_report_test'
]

from collections import OrderedDict
from copy import deepcopy
from configs.configs import ContinuousReportConfig
from utils.config_utils import config_dict2config_df, vary_config
import numpy as np
import os.path as osp

def cued_continuous_report():
    config = ContinuousReportConfig()
    
    config.experiment_name = 'cued_continuous_report'
    config.task_type = 'cued_continuous_report'
    config.num_patches = (4, )

    config_ranges = OrderedDict()
    config_ranges['cue_prob'] = [0.25, 0.5, 0.75, 1.0, ]
    config_ranges['att'] = ['feature', 'cbam', 'none', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def cued_continuous_report_set_size():
    config = ContinuousReportConfig()
    
    config.experiment_name = 'cued_continuous_report_set_size'
    config.task_type = 'cued_continuous_report'
    config.num_patches = (1, 2, 3, 4, 5, 6)

    config_ranges = OrderedDict()
    config_ranges['cue_prob'] = [1.0, ]
    config_ranges['att'] = ['cbam', 'none', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def cued_continuous_report_test():
    configs = cued_continuous_report()
    new_configs = {0: [], 1: [], 2: [], 3: []}
    for i in range(len(configs)):
        for cue in [False, True]:

            cfg: ContinuousReportConfig = deepcopy(configs.loc[i].config)
            if not cfg.freeze_cnn:
                break

            cfg.load_path = osp.join(cfg.save_path, 'net_best.pth')
            cfg.lr = 0
            cfg.max_batch = cfg.log_every = 1
            cfg.num_workers = cfg.cpu = 1
            cfg.cue_prob = cue * 1.0

            original_exp_name = cfg.experiment_name
            cfg.experiment_name = 'cued_continuous_report_test'
            cfg.save_path = cfg.save_path.replace(original_exp_name, cfg.experiment_name)
            cfg.save_path += f'_valid_cue_{cue}'

            new_configs[cfg.seed].append(cfg)

    return config_dict2config_df(new_configs)