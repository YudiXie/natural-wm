import copy

from utils.config_utils import configs_df_unpack
from configs import experiments


def target2str(targets):
    ret_str = str(targets)
    ret_str = ret_str.replace('[', '').replace(']', '')
    ret_str = ret_str.replace(',', '').replace(' ', '')
    return 'targets' + ret_str


def dms_gen_novel_images_CIFAR10_var_delay_eval():
    configs_dict = experiments.dms_gen_novel_images_CIFAR10()
    configs_list = configs_df_unpack(configs_dict)

    conditions = ['train', 'gen', 'white_noise']
    dataset_sets = ['CIFAR10', 'CIFAR10', 'WhiteNoise']
    target_sets = [[0, 1, 2, 3, 4],
                   [5, 6, 7, 8, 9],
                   None]
    delay_list = list(range(15))

    # here hard coded the last model to evaluate the last model
    # in the future should implement 'last', 'best', and 'all'
    eval_config_list = []
    for cfg in configs_list:
        for cond, dataset, target in zip(conditions, dataset_sets, target_sets):
            for delay in delay_list:
                eval_cfg = copy.deepcopy(cfg)

                eval_cfg.config_mode = 'eval'
                # the index of network to be evaluated
                eval_cfg.eval_net_num = 9000
                # the the unique condition of each evaluation
                eval_cfg.eval_name = cond + dataset + target2str(target) + str(delay)

                eval_cfg.dataset = dataset
                eval_cfg.inc_targets = target
                eval_cfg.delay_step = delay

                eval_config_list.append(eval_cfg)
    return eval_config_list


def dms_gen_novel_flatten_images_eval():
    configs_dict = experiments.dms_gen_novel_flatten_images()
    configs_list = configs_df_unpack(configs_dict)

    conditions = ['train', 'gen']
    target_sets = [[0, 1, 2, 3, 4],
                   [5, 6, 7, 8, 9]]
    net_numbers = [1000 * i_ for i_ in range(10)]

    eval_config_list = []
    for cfg in configs_list:
        for cond, target in zip(conditions, target_sets):
            for net_num in net_numbers:
                eval_cfg = copy.deepcopy(cfg)

                eval_cfg.config_mode = 'eval'
                # the index of network to be evaluated
                eval_cfg.eval_net_num = net_num
                # the the unique condition of each evaluation
                eval_cfg.eval_name = cond + target2str(target)

                cfg.inc_targets = target

                eval_config_list.append(eval_cfg)
    return eval_config_list
