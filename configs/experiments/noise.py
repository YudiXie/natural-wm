__all__ = [
    'continuous_report_multiplicative_rnn_noise_test',
    'continuous_report_additive_rnn_noise_test',
    'continuous_report_input_noise_test',
    'continuous_report_input_noise_train',
    'continuous_report_rnn_noise_train',
    'luckvogel_multiplicative_rnn_noise_test',
    'luckvogel_additive_rnn_noise_test',
    'luckvogel_input_noise_test',
    'luckvogel_input_noise_train',
    'luckvogel_rnn_noise_train',
]

from collections import OrderedDict
from copy import deepcopy
from configs.configs import ContinuousReportConfig, LuckVogelConfig
from utils.config_utils import config_dict2config_df, vary_config, add_config
from configs.experiments import continuous_report_ngym, luckvogel_ngym
import os.path as osp
    
def noise_test(config_df, new_exp_name, noise_attribute: str, noise_level_list):
    """
    Given a config_df, test with different noise levels
    """
    config_df = deepcopy(config_df)
    for i in config_df.index:
        cfg: ContinuousReportConfig = config_df.loc[i].config
        cfg.load_path = osp.join(cfg.save_path, 'net_best.pth')
        cfg.lr = 0
        cfg.max_batch = cfg.log_every = 1

    config_ranges = OrderedDict()
    config_ranges[noise_attribute] = noise_level_list
    configs = add_config(config_df, config_ranges, mode='combinatorial', new_exp_name=new_exp_name)
    return configs

def continuous_report_additive_rnn_noise_test():
    configs = continuous_report_ngym()
    return noise_test(
        configs,
        'continuous_report_additive_rnn_noise_test',
        'additive_rnn_noise',
        [0.1, 0.3, 0.5, 0.7, ]
    )

def continuous_report_multiplicative_rnn_noise_test():
    configs = continuous_report_ngym()
    return noise_test(
        configs, 
        'continuous_report_multiplicative_rnn_noise_test', 
        'rnn_noise', 
        [0, 0.1, 0.2, 0.3, ]
    )

def continuous_report_input_noise_test():
    configs = continuous_report_ngym()
    return noise_test(
        configs,
        'continuous_report_input_noise_test',
        'input_noise',
        [0.1, 0.2, 0.3, 0.4, ]
    )

def luckvogel_additive_rnn_noise_test():
    cfgs = luckvogel_ngym()
    return noise_test( 
        cfgs,
        'luckvogel_additive_rnn_noise_test',
        'additive_rnn_noise',
        [0.1, 0.3, 0.5, 0.7, ]
    )

def luckvogel_multiplicative_rnn_noise_test():
    cfgs = luckvogel_ngym()
    return noise_test(
        cfgs,
        'luckvogel_multiplicative_rnn_noise_test',
        'rnn_noise',
        [0, 0.1, 0.2, 0.3, ]
    )

def luckvogel_input_noise_test():
    cfgs = luckvogel_ngym()
    return noise_test(
        cfgs,
        'luckvogel_input_noise_test',
        'input_noise',
        [0.1, 0.2, 0.3, 0.4, ]
    )

def luckvogel_input_noise_train():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_input_noise_train'
    
    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['Classification_CIFAR10', ]
    config_ranges['input_noise'] = [0, 0.05, 0.1, 0.2, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def continuous_report_input_noise_train():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_input_noise_train'
    
    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['Classification_CIFAR10', ]
    config_ranges['input_noise'] = [0, 0.05, 0.1, 0.2, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def luckvogel_rnn_noise_train():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_rnn_noise_train'
    
    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['Classification_CIFAR10', ]
    config_ranges['additive_rnn_noise'] = [0, 0.05, 0.1, 0.2, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def continuous_report_rnn_noise_train():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_rnn_noise_train'
    
    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['Classification_CIFAR10', ]
    config_ranges['additive_rnn_noise'] = [0, 0.05, 0.1, 0.2, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs