import json
import logging
import os
import shutil
from copy import deepcopy

import numpy as np
import pandas as pd

from configs.config_global import FIG_DIR, ROOT_DIR
from configs.configs import BaseConfig


def copy_figures_from_path(source, target):
    for filename in os.listdir(source):
        path = os.path.join(source, filename)
        target_path = os.path.join(target, filename)

        if os.path.isfile(path) and (filename[-4: ] in ['.pdf', '.png', '.jpg']):
            shutil.copyfile(path, target_path)
        elif os.path.isdir(path):
            os.makedirs(target_path, exist_ok=True)
            copy_figures_from_path(path, target_path)

def copy_figures_from_experiments(configs, path):
    if isinstance(configs, dict):
        configs = config_dict2config_df(configs)
    config_list = configs_df_unpack(configs)

    path = os.path.join(FIG_DIR, path)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, config_list[0].experiment_name)
    os.makedirs(path, exist_ok=True)

    for cfg in config_list:
        cfg: BaseConfig
        sub_path = os.path.join(path, cfg.model_name)
        os.makedirs(sub_path, exist_ok=True)
        copy_figures_from_path(cfg.save_path, sub_path)

def configs_df_unpack(configs_df):
    """
    unpack configs_df to a list of configs
    args:
        configs_df: pandas.DataFrame, that contains all configs in an experiment
    return:
        config_list: unpacked list of configs
    """
    config_list = list(configs_df.loc[:, 'config'])
    return config_list


def configs_df2config_dict(config_df: pd.DataFrame):
    """
    convert configs_df to a config_dict
    this function is temporary and will be removed, it is recommended to use config_df
    to contain configs.
    config_dict is the data structure used in previous version of the codebase
    args:
        configs_df: pandas.DataFrame, that contains all configs in an experiment
    return:
        config_dict: a dictinary of configs
            the key are the random seed, and the value is a list of configs
    """
    seed_list = config_df['seed'].unique()
    config_dict = {}
    for seed in seed_list:
        config_dict[seed] = list(config_df.loc[config_df['seed'] == seed, 'config'])
    return config_dict

def config_dict2config_df(config_dict):
    seed_list = []
    config_list = []
    for seed, configs in config_dict.items():
        for config in configs:
            config_list.append(config)
            seed_list.append(seed)
    config_df = pd.DataFrame({'seed': seed_list, 'config': config_list})
    return config_df

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


def save_config(config, save_path, also_save_as_text=True):
    """
    Save config.
    adapted from https://github.com/gyyang/olfaction_evolution
    """
    if os.path.exists(save_path):
        file_path = os.path.join(save_path, 'progress.txt')
        if not config.overwrite and os.path.exists(file_path):
            try:
                exp_data = pd.read_table(file_path)
                if len(exp_data['BatchNum']) == config.max_batch // config.log_every:
                    logging.warning('Save dir {} already exists and training is already done. '.format(save_path)
                            + 'Skipped. (to overwrite, set config.overwrite = True instead)')
                    return False
            except:
                pass
        
        logging.warning('Save dir {} already exists!'.format(save_path)
                        + 'Storing info there anyway. ')
    else:
        os.makedirs(save_path)

    config_dict = config.__dict__
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f)

    if also_save_as_text:
        with open(os.path.join(save_path, 'config.txt'), "w") as f:
            for k, v in config_dict.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')
    return True


def load_config(save_path):
    """
    Load config.
    adapted from https://github.com/gyyang/olfaction_evolution
    """
    import configs.configs
    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    config = configs.configs.BaseConfig()
    for key, val in config_dict.items():
        setattr(config, key, val)
    return config


def vary_config(base_config, config_ranges, mode,
                num_seed=1, default_name=False):
    """Return configurations.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        base_config: BaseConfig object, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
        mode: str, can take 'combinatorial' or 'sequential'
        num_seed: int, number of random seeds
        default_name: bool, whether to use auto_name function

    Return:
        config_df: a pandas data frame of configs,
            each row is a config, each column is a variation of parameter
    """
    assert 'seed' not in config_ranges.keys(), 'seed cannot be specified in config range'

    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]

    attribute_dict = {}
    for key in keys:
        attribute_dict[key] = []
    attribute_dict['seed'] = []
    attribute_dict['config'] = []

    # Return combinatorial configurations,
    # config_ranges should not have repetitive values
    if mode == 'combinatorial':
        n_max = int(np.prod(dims))
        assert n_max > 0
        for seed in range(num_seed):
            for i in range(n_max):
                new_config = deepcopy(base_config)
                # Set up new config
                new_config.seed = seed
                indices = np.unravel_index(i, shape=dims)
                for key, index in zip(keys, indices):
                    val = config_ranges[key][index]
                    setattr(new_config, key, val)
                    attribute_dict[key].append(val)

                attribute_dict['seed'].append(seed)
                attribute_dict['config'].append(new_config)

    # Return sequential configurations.
    # config_ranges values should have equal length,
    # otherwise this will only loop through the shortest one
    elif mode == 'sequential':
        n_max = np.min(dims)
        assert n_max > 0
        for seed in range(num_seed):
            for i in range(n_max):
                new_config = deepcopy(base_config)
                # Set up new config
                new_config.seed = seed
                for key in keys:
                    val = config_ranges[key][i]
                    setattr(new_config, key, val)
                    attribute_dict[key].append(val)

                attribute_dict['seed'].append(seed)
                attribute_dict['config'].append(new_config)
    else:
        raise ValueError('Unknown mode {}'.format(str(mode)))

    configs_df = pd.DataFrame(attribute_dict)

    for i, row in configs_df.iterrows():
        config = row.loc['config']
        if default_name:
            config.model_name = str(i).zfill(6)
        else:
            name = 'model'
            for key in keys:
                name += '_' + str(key) + str(row.loc[key])

            # replace char that are not suitable for path
            name = name.replace(",", "").replace(" ", "_")
            name = name.replace("[", "_").replace("]", "_").replace(".", "_")
            name = name.replace("'", "")
            config.model_name = name + '_s' + str(row.loc['seed'])

        config.save_path = os.path.join(ROOT_DIR, 'experiments',
                                        config.experiment_name,
                                        config.model_name)

    return configs_df

def add_config(base_config_df, config_ranges, mode='combinatorial', new_exp_name=None):
    """
    Exapnd an existing experiment by adding new configs
    """

    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]

    attribute_dict = {}
    for key in keys:
        attribute_dict[key] = []
    attribute_dict['seed'] = []
    attribute_dict['config'] = []

    # Return combinatorial configurations
    if mode == 'combinatorial':
        n_max = int(np.prod(dims))
        assert n_max > 0
        for idx in base_config_df.index:
            for i in range(n_max):
                new_config = deepcopy(base_config_df.loc[idx].config)
                # Set up new config
                indices = np.unravel_index(i, shape=dims)
                for key, index in zip(keys, indices):
                    val = config_ranges[key][index]
                    setattr(new_config, key, val)
                    attribute_dict[key].append(val)
                
                if new_exp_name is not None:
                    new_config.experiment_name = new_exp_name

                attribute_dict['seed'].append(new_config.seed)
                attribute_dict['config'].append(new_config)
    elif mode == 'sequential':
        n_max = np.min(dims)
        assert n_max > 0
        for idx in base_config_df.index:
            for i in range(n_max):
                new_config = deepcopy(base_config_df.loc[idx].config)
                # Set up new config
                for key in keys:
                    val = config_ranges[key][i]
                    setattr(new_config, key, val)
                    attribute_dict[key].append(val)
                
                if new_exp_name is not None:
                    new_config.experiment_name = new_exp_name

                attribute_dict['seed'].append(new_config.seed)
                attribute_dict['config'].append(new_config)
    else:
        raise ValueError('Unknown mode {}'.format(str(mode)))

    configs_df = pd.DataFrame(attribute_dict)

    for i, row in configs_df.iterrows():
        config = row.loc['config']
        assert config.model_name[-3: -1] == '_s'
        name = config.model_name[: -3]

        for key in keys:
            name += '_' + str(key) + str(row.loc[key])

        # replace char that are not suitable for path
        name = name.replace(",", "").replace(" ", "_")
        name = name.replace("[", "_").replace("]", "_").replace(".", "_")
        name = name.replace("'", "")
        config.model_name = name + '_s' + str(row.loc['seed'])
        config.save_path = os.path.join(ROOT_DIR, 'experiments',
                                        config.experiment_name,
                                        config.model_name)

    return configs_df