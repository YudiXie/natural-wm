__all__ = [
    'classification_pretrain_cnn_size',
    'classification_pretrain_cnn_norm', 
    'contrastive_pretrain',
    'classification_pretrain_mnist',
    'luckvogel_classification_pretrain', 
    'luckvogel_contrastive_pretrain',
]

from collections import OrderedDict
from configs.configs import (LuckVogelConfig, BaseConfig, )
from utils.config_utils import (config_dict2config_df, configs_df2config_dict,
                                vary_config)

def set_classification_hyperparams(config: BaseConfig):

    config.wdecay = 1e-3
    config.batch_size = 128
    config.optimizer_type = 'SGD'
    config.log_every = 200

    config.task_type = 'classification'
    config.use_lr_scheduler = True
    config.scheduler_type = 'ExponentialLR'

    config.model_type = 'ResNet'
    config.neurogym = False

    return config

def classification_pretrain_cnn_size():
    config = set_classification_hyperparams(BaseConfig())

    config.experiment_name = 'classification_pretrain'
    config.dataset = 'CIFAR10'
    config.model_class_size = 10

    config_ranges = OrderedDict()
    config_ranges['cnn_norm'] = ['instancenorm', 
                                 'layernorm', ]
    config_ranges['cnn_width'] = [4, 8, 16, 32, 128, 256, 512, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    return configs

def classification_pretrain_cnn_norm():
    config = set_classification_hyperparams(BaseConfig())

    config.experiment_name = 'classification_pretrain'
    config.dataset = 'CIFAR10'
    config.model_class_size = 10

    config_ranges = OrderedDict()
    config_ranges['cnn_norm'] = ['none',
                                 'batchnorm', 
                                 'instancenorm', 
                                 'instancenorm_affine',
                                 'layernorm',
                                 'groupnorm']
    config_ranges['cnn_width'] = [64, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    return configs

def classification_pretrain_mnist():
    config = set_classification_hyperparams(BaseConfig())

    config.experiment_name = 'classification_pretrain_mnist'
    config.dataset = 'MNIST-Colored'
    config.model_class_size = 10

    config_ranges = OrderedDict()
    config_ranges['cnn_norm'] = ['batchnorm', 
                                 'instancenorm', 
                                 'layernorm', ]
    config_ranges['cnn_width'] = [4, 8, 16, 32, 64, 128, 256, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    return configs

def luckvogel_classification_pretrain():
    config = set_classification_hyperparams(LuckVogelConfig())

    config.experiment_name = 'luckvogel_classification_pretrain'
    config.dataset = 'LuckVogelClassification'
    config.model_class_size = 12
    config.num_patches = list(range(1, 13))
    config.max_batch = 2000

    config_ranges = OrderedDict()
    config_ranges['cnn_norm'] = ['instancenorm', 'batchnorm', ]
    config_ranges['cnn_width'] = [64, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def contrastive_pretrain():
    config = BaseConfig()
    config.experiment_name = 'contrastive_pretrain'
    config.model_type = 'ResNetSimCLR'
    config.task_type = 'contrastive_learning'
    config.dataset = 'CIFAR10-SimCLR'

    config.max_batch = 30000
    config.log_every = 200

    config.perform_test = False
    config.batch_size = 128
    config.wdecay = 1e-4
    config.early_stop = False
    config.neurogym = False

    config.cpu = 4
    config.num_workers = 4
    config.hours = 48
    config.gpu_constraint = 'ampere'

    config_ranges = OrderedDict()
    config_ranges['cnn_norm'] = ['instancenorm', 'layernorm', ]
    config_ranges['cnn_width'] = [64, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs


def luckvogel_contrastive_pretrain():
    config = LuckVogelConfig()
    config.experiment_name = 'luckvogel_contrastive_pretrain'
    config.model_type = 'ResNetSimCLR'
    config.task_type = 'contrastive_learning'
    config.dataset = 'LuckVogel-SimCLR'

    config.max_batch = 4000
    config.log_every = 50
    config.perform_test = False
    config.batch_size = 256
    config.wdecay = 1e-4
    config.early_stop = False
    config.neurogym = False

    config_ranges = OrderedDict()
    config_ranges['cnn_norm'] = ['layernorm', ]
    config_ranges['cnn_width'] = [64, ]
    config_ranges['max_pretrain_set_size'] = [1, 2, 4, 8, 12, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    for i in range(len(configs)):
        cfg: LuckVogelConfig = configs.loc[i].config
        cfg.num_patches = list(range(1, 1 + cfg.max_pretrain_set_size))

    return configs