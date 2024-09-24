from collections import OrderedDict
from copy import deepcopy
from configs.configs import SequentialContinuousReportConfig
from utils.config_utils import config_dict2config_df, vary_config
import os.path as osp

__all__ = [
    'sequential_continuous_report',
    'sequential_continuous_report_noatt',
    'sequential_continuous_report_rnn_size',
    'sequential_continuous_report_rnn_size_end2end',
]

def sequential_continuous_report():    
    config = SequentialContinuousReportConfig()
    config.experiment_name = 'sequential_continuous_report'

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['none', 'Classification_CIFAR10', ]
    config_ranges['freeze_cnn'] = [False, True, ]

    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)
    return configs

def sequential_continuous_report_noatt():
    config = SequentialContinuousReportConfig()
    config.experiment_name = 'sequential_continuous_report_noatt'
    config.att = 'none'

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['none', 'Classification_CIFAR10', ]
    config_ranges['freeze_cnn'] = [False, True, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def sequential_continuous_report_rnn_size():
    config = SequentialContinuousReportConfig()
    config.experiment_name = 'sequential_continuous_report_rnn_size'
    config.task_type = 'sequential_continuous_report'
    config.gpu_constraint = '18GB'

    config_ranges = OrderedDict()
    config_ranges['hidden_size'] = [32, 64, 128, 256, 512, 1024]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def sequential_continuous_report_rnn_size_end2end():
    config = SequentialContinuousReportConfig()
    config.experiment_name = 'sequential_continuous_report_rnn_size_end2end'
    config.task_type = 'sequential_continuous_report'
    config.gpu_constraint = '18GB'
    config.cnn_pret = 'none'
    config.freeze_cnn = False

    config_ranges = OrderedDict()
    config_ranges['hidden_size'] = [32, 1024, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs