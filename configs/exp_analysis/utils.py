import torch
import numpy as np
import os.path as osp
import pandas as pd
import analysis.plots as plots
from configs.configs import ContinuousReportConfig
from matplotlib import cm
from copy import deepcopy

from scipy.stats import circstd
from utils.config_utils import configs_df2config_dict
from typing import List

__all__ = [
    'get_luck_vogel_performance', 
    'get_learning_curve',
    'get_continuous_report_performance_error',
    'get_continuous_report_nontarget_error',
    'continuous_report_model_size_analysis'
]

def get_luck_vogel_performance(cfgs, idx, key='acc'):
    """
    retrieve the performance from training results.
    :args:
        cfgs: a config dataframe
        idx: the index to retrieve from
        key: the information to retrieve, should be chose from ['acc', 'hit_rate', 'false_alarm']
    :returns:
        2D list of shape (len(set_sizes), num_seeds), the accuracy (or other metrics) for each set size
    """

    cfgs = configs_df2config_dict(cfgs)
    num_seeds = len(cfgs)
    set_sizes = cfgs[0][idx].num_patches
    performance = [[] for _ in set_sizes]

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]
        try:
            file_path = osp.join(cfg.save_path, 'progress.txt')
            exp_data = pd.read_table(file_path)

            if len(exp_data['test/TestLoss']) < cfg.max_batch // cfg.log_every: 
                print('Incomplete data:')
                raise ValueError(exp_data)
            best_idx = exp_data['test/TestLoss'].argmin()

            for j, size in enumerate(set_sizes):
                acc = exp_data[f'test/{key}_{size}'][best_idx]
                performance[j].append(acc)
            
        except:
            print('Error occur when reading data at', cfg.save_path)

    return performance

def get_learning_curve(cfgs, idx, key='TestLoss'):
    """
    retrieve the learning curve from training results.
    """

    cfgs = configs_df2config_dict(cfgs)
    num_seeds = len(cfgs)
    t_steps = cfgs[0][idx].max_batch // cfgs[0][idx].log_every
    performance = [[] for _ in range(t_steps)]

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]
        try:
            file_path = osp.join(cfg.save_path, 'progress.txt')
            exp_data = pd.read_table(file_path)

            if len(exp_data['test/TestLoss']) < t_steps: 
                print('Incomplete data:')
                raise ValueError(exp_data)

            for step in range(t_steps):
                acc = exp_data['test/' + key][step]
                performance[step].append(acc)
            
        except:
            print('Error occur when reading data at', cfg.save_path)

    return performance

def get_continuous_report_performance_error(cfgs, idx, detail=False):
    """
    retrieve the performance and error distribution from training results.
    :args:
        cfgs: a config dataframe
        idx: the index to retrieve from
        detail: for the sequential continuous report task, setting
            detail=True would return performance of each rank
    :returns: a tuple
        performance: 2D list of shape (len(set_sizes), num_seeds),
            the recall SD for each set size (in deg)
        errors: 3D list of shape (len(set_sizes), num_seeds, num_trials),
            the error on each trial.
            Note that if cfg.output_uncertainty = True, each element in
            the 3D list would be a tuple (error, uncertainty) rather than
            a single float
        precision: 2D list of shape (len(set_sizes), num_seeds), the
            precision for each set size (in rad)
        detailed_performance, (if detail = True): 3D list of shape
            (len(set_sizes), set_size, num_seeds), the performance of
            each rank
    """

    cfgs = configs_df2config_dict(deepcopy(cfgs))
    num_seeds = len(cfgs)
    set_sizes = cfgs[0][idx].num_patches

    performance = [[] for _ in set_sizes]
    precisions = [[] for _ in set_sizes]
    errors = [[] for _ in set_sizes]

    if detail:
        detailed_performance = [[[] for rank in range(n)] for n in set_sizes]

    for seed in range(num_seeds):
        cfg: ContinuousReportConfig = cfgs[seed][idx]

        file_path = osp.join(cfg.save_path, 'progress.txt')

        try:
            exp_data = pd.read_table(file_path)
            if len(exp_data['test/TestLoss']) < cfg.max_batch // cfg.log_every: 
                print('Incomplete data:', file_path)
                raise ValueError
            
            best_idx = exp_data['test/TestLoss'][: cfg.max_batch // cfg.log_every].argmin()

            for j, size in enumerate(set_sizes):
                error = torch.load(osp.join(cfg.save_path, f'error_{size}.pth'))
                if len(error) == 0:
                    print('Incomplete data:', file_path)
                    raise ValueError(error)

                if cfg.output_uncertainty:
                    uncertainty = torch.load(osp.join(cfg.save_path, f'uncertainty_{size}.pth'))
                    assert len(error) == len(uncertainty)
                    errors[j].append(list(zip(error, uncertainty)))
                else:
                    errors[j].append(error)

                # load SD
                # recall_std = exp_data[f'error_std_{size}'][best_idx]
                # Calculate SD and precision
                recall_std = circstd(error, low=-180, high=180)
                chance_std = circstd(np.random.rand(len(error)) * 360 - 180, low=-180, high=180)

                performance[j].append(recall_std)
                precision = (1 / recall_std - 1 / chance_std) * 180 / np.pi
                precisions[j].append(precision)

            if detail:
                for j, size in enumerate(set_sizes):
                    for rank in range(size):
                        recall_std = exp_data[f'test/error_std_{size}_{rank}'][best_idx]
                        detailed_performance[j][rank].append(recall_std)
        
        except:
            print('Error occur when reading data at', cfg.save_path)

    if detail:
        return performance, errors, precisions, detailed_performance
    else:
        return performance, errors, precisions

def get_continuous_report_nontarget_error(cfgs, idx):

    cfgs = configs_df2config_dict(deepcopy(cfgs))
    num_seeds = len(cfgs)
    set_sizes = cfgs[0][idx].num_patches
    errors = [[] for _ in set_sizes]

    for seed in range(num_seeds):
        cfg: ContinuousReportConfig = cfgs[seed][idx]

        for j, size in enumerate(set_sizes):
            if size == 1:
                continue
            error = torch.load(osp.join(cfg.save_path, f'nontarget_error_{size}.pth'))
            if len(error) == 0:
                print('Incomplete data:', cfg.save_path)
                raise ValueError(error)
            
            errors[j].append(error)

    return errors

def continuous_report_model_size_analysis(
    cfgs, idx: int,
    model_size_list: List[int],
    fig_dir: str, fig_prefix: str,
    cmap=cm.get_cmap('coolwarm'),
    set_sizes=[1, 2, 3, 4, 5, 6],
    plot_model_size_idx='all',
    plot_set_size_idx=[1, 3, 5],
    module='RNN',
    legend_fontsize=10,
    SD_vs_setsize_kwargs={},
    SD_vs_modelsize_kwargs={},
    precision_vs_modelsize_kwargs={},
):

    if plot_model_size_idx == 'all':
        plot_model_size_idx = list(range(len(model_size_list)))
    plot_model_size = [model_size_list[i] for i in plot_model_size_idx]
    if plot_set_size_idx == 'all':
        plot_set_size_idx = list(range(len(set_sizes)))
    plot_set_size = [set_sizes[i] for i in plot_set_size_idx]

    color_list = [cmap(i) for i in np.linspace(0.4, 0, len(model_size_list))]
    performance = {} # SD
    _precision = {} # precision

    for i, model in enumerate(model_size_list):
        performance[model], errors, _precision[model] = get_continuous_report_performance_error(cfgs, idx * len(model_size_list) + i)

    plots.continuous_report_plot(
        performance, 
        set_sizes, 
        plot_model_size,
        f'{fig_prefix}_recallSDvsSetSize_compare_{module}_size',
        plot_dir=fig_dir,
        legend_title=f'{module} size',
        colors=color_list,
        legend_fontsize=legend_fontsize,
        **SD_vs_setsize_kwargs
    )

    color_list = [cmap(i) for i in np.linspace(1, 0.6, len(plot_set_size))]    
    recall_std = [[performance[model][i] for model in model_size_list] for i in plot_set_size_idx]
    precision = [[_precision[model][i] for model in model_size_list] for i in plot_set_size_idx]

    plots.error_plot(
        np.log2(model_size_list),
        recall_std,
        f'log2({module} size)',
        y_label='Recall std (deg)',
        label_list=plot_set_size,
        fig_dir=fig_dir,
        fig_name=f'{fig_prefix}_log_{module}_size_vs_SD',
        legend_title='Set size',
        colors=color_list,
        legend_fontsize=legend_fontsize,
        **SD_vs_modelsize_kwargs
    )

    plots.error_plot(
        model_size_list,
        recall_std,
        f'{module} size',
        y_label='Recall std (deg)',
        label_list=plot_set_size,
        fig_dir=fig_dir,
        fig_name=f'{fig_prefix}_{module}_size_vs_SD',
        legend_title='Set size',
        colors=color_list,
        legend_fontsize=legend_fontsize,
        **SD_vs_modelsize_kwargs
    )

    plots.error_plot(
        np.log2(model_size_list),
        precision,
        f'log2({module} size)',
        y_label='Precision (1 / rad)',
        label_list=plot_set_size,
        fig_dir=fig_dir,
        fig_name=f'{fig_prefix}_log_{module}_size_vs_precision',
        legend_title='Set size',
        colors=color_list,
        legend_fontsize=legend_fontsize,
        **precision_vs_modelsize_kwargs
    )

    plots.error_plot(
        model_size_list,
        precision,
        f'{module} size',
        y_label='Precision (1 / rad)',
        label_list=plot_set_size,
        fig_dir=fig_dir,
        fig_name=f'{fig_prefix}_{module}_size_vs_precision',
        legend_title='Set size',
        colors=color_list,
        legend_fontsize=legend_fontsize,
        **precision_vs_modelsize_kwargs
    )