import torch
import numpy as np
import os.path as osp
import pandas as pd
import analysis.plots as plots
from configs.configs import ContinuousReportConfig
import configs.experiments as experiments
from matplotlib import cm
from copy import deepcopy

from scipy.stats import circstd
from utils.config_utils import configs_df2config_dict
from configs.exp_analysis.utils import *
from typing import List

__all__ = [
    # Experiments
    'sequential_continuous_report_analysis',
    'sequential_continuous_report_rnn_size_analysis',
    # Not used
    'get_continuous_report_gradient_info',
    'continuous_report_gradient_analysis',
    'sequential_continuous_report_gradient_analysis'
]

def sequential_continuous_report_analysis():
    cfgs = experiments.sequential_continuous_report()

    model_list = ['ResNet', 'pretResNet', ]
    set_sizes = [1, 2, 3, 4, 5, 6]
    plot_set_sizes = [1, 2, 4, 6]
    yticks_list = [[0.0, 0.05, 0.10], [0.0, 0.02, 0.04], [0.0, 0.05, 0.10]]
    cmap = cm.get_cmap('coolwarm')
    dist_colors_list = [
        [cmap(i) for i in np.linspace(0.0, 0.4, len(plot_set_sizes))],
        [cmap(1 - i) for i in np.linspace(0.0, 0.4, len(plot_set_sizes))],
        [cmap(i) for i in np.linspace(0.0, 0.4, len(plot_set_sizes))],
    ]

    performance = {}
    precision = {}
    learning_curves = []

    for i, model in enumerate(model_list):
        performance[model], errors, precision[model], detailed_performance = \
            get_continuous_report_performance_error(cfgs, i, detail=True)
        learning_curves.append(get_learning_curve(cfgs, i))

        # plot error distribution at different set sizes
        plots.error_distribution_plot(
            [errors[n - 1] for n in plot_set_sizes],
            plot_set_sizes,
            -120, 120, 5,
            f'{model}_error_dist',
            legend_title='Set Size',
            yticks=yticks_list[i],
            colors=dist_colors_list[i],
            plot_dir='seq_continuous_report',
        )

        # plot Precision with fitted curve
        plots.continuous_report_plot(
            precision,
            set_sizes,
            [model, ],
            f'{model}_precision',
            y_label='Precision (1 / rad)',
            colors=[(1, 127 / 255, 14 / 255), ] if i > 0 else None,
            plot_dir='seq_continuous_report',
        )

        plots.sequential_performance_plot(
            detailed_performance,
            set_sizes,
            f'{model}_detail_performance',
            'seq_continuous_report',
            plot_reciprocal=False,
            legend_title='Set Size'
        )

    # plot how error sd changes with set size for two models
    plots.continuous_report_plot(
        performance,
        set_sizes,
        ['ResNet', 'pretResNet', ],
        'recallSDvsSetSize_pret_vs_nonpret',
        legend_loc='upper left',
        legend_fontsize=11,
        colors=['#5AA9E6', '#FF6392'],
        label_list=['End-to-end', 'Pre-trained'],
        plot_dir='seq_continuous_report',
    )

    cfg = cfgs.loc[0, 'config']
    x_axis = np.arange(cfg.log_every, cfg.max_batch + 1, cfg.log_every)

    # plot learning curve for two models
    plots.error_plot(
        x_axis,
        learning_curves, 
        'Training Step',
        'Test Loss',
        fig_name=f'LearningCurve_pret_vs_nonpret',
        fig_dir='seq_continuous_report',
        colors=['#5AA9E6', '#FF6392'],
        label_list=['End-to-end', 'Pre-trained'],
        mode='errorshade',
        xticks=[0, 20000, 40000],
    )

def sequential_continuous_report_rnn_size_analysis():
    continuous_report_model_size_analysis(
        experiments.sequential_continuous_report_rnn_size(), 0,
        [32, 64, 128, 256, 512, 1024],
        'seq_continuous_report_model_size',
        'pretrained',
    )

def get_continuous_report_gradient_info(cfgs, idx):

    num_seeds = len(cfgs)
    set_sizes = cfgs[0][idx].num_patches

    dist_grads_abs_mean = [[[] for i in range(20)] for _ in set_sizes]
    dist_grads_mean = [[[] for i in range(20)] for _ in set_sizes]

    target_grads_dist = [[] for _ in set_sizes]
    spurious_grads_dist = [[] for _ in range(len(set_sizes) - 1)]

    target_grads_mean = [[] for _ in set_sizes]
    spurious_grads_mean = [[] for _ in range(len(set_sizes) - 1)]

    grad_ratio_list = [[] for _ in set_sizes]
    errors = [[] for _ in set_sizes]
    mind_std = [[[] for d in range(6, 16)] for _ in set_sizes]

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        #try:

        file_path = osp.join(cfg.save_path, 'grad.pth')
        target_grads, other_grads, grad_ratio, dist_grads, nt_errors, mind_errors = torch.load(file_path)

        for i, n in enumerate(set_sizes):

            if n > 1:
                for d in range(20):
                    data = np.array(dist_grads[i][d])
                    if len(data) > 0:
                        dist_grads_mean[i][d].append(data.mean())
                        dist_grads_abs_mean[i][d].append(np.abs(data).mean())

                for d in range(6, 16):
                    data = np.array(mind_errors[i][d - 6])
                    mind_std[i][d - 6].append(circstd(data, low=-180, high=180))

                spurious_grads_dist[i - 1].append([])
                for d, grads in enumerate(dist_grads[i]):
                    spurious_grads_dist[i - 1][-1] += grads

                spurious_grads_mean[i - 1].append(np.mean(spurious_grads_dist[i - 1][-1]))

            target_grads_dist[i].append(target_grads[i])
            target_grads_mean[i].append(np.mean(target_grads[i]))
            
            grad_ratio_list[i].append(1 - np.mean(grad_ratio[i]))
            errors[i].append(np.array(nt_errors[i]))
            
        #except:
        #    print('Error occur when reading data at', cfg.save_path)

    return (target_grads_dist, spurious_grads_dist), \
           (target_grads_mean, spurious_grads_mean), \
           (dist_grads_mean, dist_grads_abs_mean), grad_ratio_list, errors, mind_std

def continuous_report_gradient_analysis(dir_name = 'continuous_report'):
    cfgs = experiments.continuous_report_gradient_analysis()

    model_list = ['ResNet', 'pretResNet', ]
    set_sizes = [1, 2, 3, 4, 5, 6]

    grad_ratios = []
    target_grad_mean = []
    spurious_grad_mean = []

    for i, model in enumerate(model_list):
        ret = get_continuous_report_gradient_info(cfgs, i)
        
        grad_ratios.append(ret[3])
        target_grad_mean.append(ret[1][0])
        spurious_grad_mean.append(ret[1][1])

        dist_grads_mean, dist_grads_abs_mean = ret[2]
        plot_sizes = [2, 4, 6]

        for data, name in ((dist_grads_mean, ''), (dist_grads_abs_mean, '_abs')):

            plot_data = []
            for idx, n in enumerate(set_sizes):
                if n in plot_sizes:
                    plot_data.append(data[idx][6: 20])

            plots.error_plot(
                np.arange(6, 20),
                plot_data,
                'Distance to Target',
                'Mean abs(Gradient)' if name == '_abs' else 'Gradient',
                plot_sizes,
                fig_dir=f'{dir_name}_gradient',
                fig_name=f'grad_distance{name}_{model}',
                legend_title='Set Size',
                xticks=[6, 12, 18]
            )

        for data, name, label in (
            ((ret[0][0][0], ret[0][0][1], ret[0][0][3], ret[0][0][5]), 'target', ['N = 1', 'N = 2', 'N = 4', 'N = 6']), 
            ((ret[0][1][0], ret[0][1][2], ret[0][1][4]), 'spurious', ['N = 2', 'N = 4', 'N = 6'])
        ):

            plots.error_distribution_plot(
                data,
                label,
                -2, 2, 0.2,
                plot_dir=f'{dir_name}_gradient',
                plot_name=f'grad_distribution_{name}_{model}',
                x_label='Gradient'
            )

        plots.error_distribution_plot(
            [ret[4][1], ret[4][3], ret[4][5]],
            ['N = 2', 'N = 4', 'N = 6'],
            -168, 168, 24,
            f'{model}_distribution_nt',
            plot_dir=dir_name
        )

        plots.error_plot(
            np.arange(6, 16),
            ret[5][1: ],
            'Distance to Target',
            'Recall std',
            [2, 3, 4, 5, 6],
            fig_dir=dir_name,
            fig_name=f'dist_error_{model}',
            xticks=[5, 10, 15, ],
            legend_title='Set Size'
        )

    plots.continuous_report_gradient_ratio_plot(
        set_sizes,
        grad_ratios,
        model_list
    )

    for data, name, x_axis in (
        (target_grad_mean, 'target', set_sizes), 
        (spurious_grad_mean, 'spurious', set_sizes[1: ])
    ):

        plots.error_plot(
            x_axis,
            data,
            'Set Size',
            'Mean Gradient',
            model_list,
            fig_dir=f'{dir_name}_gradient',
            fig_name=f'grad_setsize_{name}'
        )

def sequential_continuous_report_gradient_analysis():
    cfgs = experiments.sequential_continuous_report_gradient_analysis()
    continuous_report_gradient_analysis(cfgs, 'seq_continuous_report')