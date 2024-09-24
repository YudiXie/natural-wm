__all__ = [
    'continuous_report_demo',
    'continuous_report_large',
    'continuous_report_ngym',
    'continuous_report_noatt',
    'continuous_report_noatt_norm',
    'continuous_report_unfreeze',
    'continuous_report_with_uncertainty',
    'continuous_report_compare_norm',
    'continuous_report_diff_cnn',
    'continuous_report_rnn_size',
    'continuous_report_cnn_size',
    'continuous_report_rnn_size_end2end',
    'continuous_report_cnn_size_end2end',
]

from collections import OrderedDict
from copy import deepcopy
from configs.configs import ContinuousReportConfig
from utils.config_utils import config_dict2config_df, vary_config, add_config
import os.path as osp

def continuous_report_demo():
    config = ContinuousReportConfig()

    config.experiment_name = 'continuous_report_demo'
    config.input_resolution = (224, 224)
    config.att = 'none'
    config.cnn_archi = 'ResNet-18'
    config.max_batch = 1
    config.log_every = 1
    config.test_batch = 1

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['Classification_ImageNet', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    return configs

def continuous_report_ngym():
    config = ContinuousReportConfig()

    config.experiment_name = 'continuous_report_ngym'
    config.task_type = 'continuous_report'

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['none', 'Classification_CIFAR10']
    config_ranges['freeze_cnn'] = [False, True]

    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)
    return configs

def continuous_report_unfreeze():
    config = ContinuousReportConfig()

    config.experiment_name = 'continuous_report_unfreeze'
    config.task_type = 'continuous_report'
    config.freeze_cnn = False
    config.log_every = 100
    config.max_batch = 1000

    config_ranges = OrderedDict()
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    old_configs = continuous_report_ngym()
    old_configs = old_configs.loc[old_configs['freeze_cnn'] == True]
    for seed in range(4):
        cfg = configs.iloc[seed].config
        old_cfg = old_configs.iloc[seed].config
        assert cfg.seed == old_cfg.seed
        cfg.load_path = osp.join(old_cfg.save_path, 'net_best.pth')

    return configs

def continuous_report_large():
    config = ContinuousReportConfig()

    config.experiment_name = 'continuous_report_large'
    config.input_resolution = (224, 224)
    config.gpu_constraint = '18GB'
    config.att = 'none'
    config.cnn_archi = 'ResNet-18'

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['R3M', 'Classification_ImageNet', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def continuous_report_noatt():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_noatt'
    config.att = 'none'

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['none', 'Classification_CIFAR10']
    config_ranges['freeze_cnn'] = [False, True]

    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)
    return configs

def continuous_report_noatt_norm():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_noatt_norm'
    config.att = 'none'
    config.gpu_constraint = '18GB'

    config_ranges = OrderedDict()
    config_ranges['cnn_pret'] = ['none', 'Classification_CIFAR10']
    config_ranges['freeze_cnn'] = [False, True]
    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)

    config_ranges = OrderedDict()
    config_ranges['cnn_norm'] = ['instancenorm_affine', 'layernorm']
    configs = add_config(configs, config_ranges, mode='sequential')

    return configs

def continuous_report_diff_cnn():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_diff_cnn'

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

def continuous_report_with_uncertainty():
    config = ContinuousReportConfig()

    config.experiment_name = 'continuous_report_with_uncertainty'
    config.task_type = 'continuous_report'
    config.output_uncertainty = True
    config.num_patches = (6, )

    config_ranges = OrderedDict()

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def continuous_report_compare_norm():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_compare_norm'
    config.task_type = 'continuous_report'

    config_ranges = OrderedDict()
    config_ranges['cnn_norm'] = ['groupnorm', 'instancenorm', 'batchnorm', 'layernorm', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def continuous_report_rnn_size():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_rnn_size'
    config.task_type = 'continuous_report'

    config_ranges = OrderedDict()
    config_ranges['hidden_size'] = [32, 64, 128, 256, 512, 1024]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def continuous_report_cnn_size():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_cnn_size'
    config.task_type = 'continuous_report'

    config_ranges = OrderedDict()
    config_ranges['cnn_width'] = [8, 16, 32, 64, 128, 256]
    config_ranges['gpu_constraint'] = ['high-capacity'] * 5 + ['18GB']

    configs = vary_config(config, config_ranges, mode='sequential', num_seed=4)
    return configs

def continuous_report_rnn_size_end2end():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_rnn_size_end2end'
    config.task_type = 'continuous_report'
    config.freeze_cnn = False
    config.cnn_pret = 'none'

    config_ranges = OrderedDict()
    config_ranges['hidden_size'] = [4, 8, 16, 32, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def continuous_report_cnn_size_end2end():
    config = ContinuousReportConfig()
    config.experiment_name = 'continuous_report_cnn_size_end2end'
    config.task_type = 'continuous_report'
    config.freeze_cnn = False
    config.cnn_pret = 'none'

    config_ranges = OrderedDict()
    config_ranges['cnn_width'] = [4, 8, 16, 32, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs