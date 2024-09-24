import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.plots import adjust_figure

from configs.config_global import FIG_DIR
from utils.general_utils import check_run_complete


plt.rcParams.update({'font.size': 16})
line_styles = ['-', '--', ':']


if not osp.exists(FIG_DIR):
    os.makedirs(FIG_DIR)


def get_performance(config, perf_key: str = 'TrainAcc'):
    """ Get the logged performance of a model on a task. """
    if not check_run_complete(config):
        last_5_acc = np.nan
    else:
        file_path = osp.join(config.save_path, 'progress.txt')
        exp_data = pd.read_table(file_path)
        last_5_acc = np.array(exp_data[perf_key][-5:]).mean()
    
    return last_5_acc


def error_bar_2par(config_df,
                   exp_name,
                   fig_name,
                   chance_level=0.5,
                   legend_title=None,
                   x_label=None,
                   y_label=None,
                   fig_title=None,
                   perf_key='TrainAcc',
                   hline=True):
    """
        make an error bar plot of the performance from a config_df
        that has two varying parameters and different random seeds.
        each first variable parameter dimension is one line + error bar
        second variable parameter dimension is the x axis

        args:
            config_df: a dataframe that only vary in the first two dimensions and random seeds
                with columns ['var1', 'var2', 'seed', 'config']
                var1: is a categorical variable
                var2: is a float or int variable
    """
    assert len(config_df.columns) == 4 \
           and config_df.columns[2] == 'seed' \
           and config_df.columns[3] == 'config'

    # get performance from log
    performance_list = [get_performance(cfg, perf_key) for cfg in config_df.loc[:, 'config']]
    config_df['performance'] = performance_list

    # group by the first 2 columns, this average performance across seeds
    groupby_obj = config_df.groupby(list(config_df.columns)[:2])
    reduced_df = groupby_obj.mean()
    reduced_df['mean'] = reduced_df['performance']
    reduced_df['std'] = groupby_obj.std(ddof=0)['performance']

    groups = list(reduced_df.index.levels[0])
    x_axis = list(reduced_df.index.levels[1])

    plt.figure(figsize=(6, 5))
    for g in groups:
        plt.errorbar(
            x_axis,
            reduced_df.loc[(g,), 'mean'],
            reduced_df.loc[(g,), 'std'],
            label=g,
            linewidth=2,
            capsize=4,
            capthick=2
        )

    if hline:
        plt.hlines(
            chance_level, np.min(x_axis), np.max(x_axis),
            label='Chance Level',
            linestyle='dotted', color='r',
            linewidth=2
        )

    if legend_title is None:
        legend_title = reduced_df.index.names[0]
    plt.legend(title=legend_title)

    if x_label is None:
        x_label = reduced_df.index.names[1]
    plt.xlabel(x_label)

    if y_label is None:
        y_label = 'performance'
    plt.ylabel(y_label)

    plt.title(fig_title)

    adjust_figure()
    os.makedirs(osp.join(FIG_DIR, exp_name), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, exp_name, fig_name + '.pdf'), transparent=True)
    plt.close()


def bar_2par(config_df,
             exp_name,
             fig_name,
             chance_level=0.5,
             legend_title=None,
             x_label=None,
             y_label=None,
             fig_title=None,
             perf_key='TrainAcc',
             ylim=[0, 1.5],
             hline=True,
             bar_label=True,
             legend_fontsize=8):
    """
        make a bar plot of the performance from a config_df
        that has two varying parameters and different random seeds.
        each first variable parameter dimension is a group of bars with the same color
        second variable parameters are different bars on the x axis

        args:
            config_df: a dataframe that only vary in the first two dimensions and random seeds
                with columns ['var1', 'var2', 'seed', 'config']
                var1: is a categorical variable
                var2: is a categorical variable
    """
    assert len(config_df.columns) == 4 \
           and config_df.columns[2] == 'seed' \
           and config_df.columns[3] == 'config'

    # get performance from log
    performance_list = [get_performance(cfg, perf_key) for cfg in config_df.loc[:, 'config']]
    config_df['performance'] = performance_list

    # group by the first 2 columns, this average performance across seeds
    groupby_obj = config_df.groupby(list(config_df.columns)[:2])
    reduced_df = groupby_obj.mean()
    reduced_df['mean'] = reduced_df['performance']
    reduced_df['std'] = groupby_obj.std(ddof=0)['performance']

    groups = list(reduced_df.index.levels[0])
    num_groups = len(groups)
    width = 0.7 / num_groups  # the width of the bars

    x_axis_labels = list(reduced_df.index.levels[1])
    x_axis = np.arange(len(x_axis_labels))  # the label locations

    fig, ax = plt.subplots()
    for i, g in enumerate(groups):
        offset = i * width - ((num_groups - 1) * width / 2)
        rect = ax.bar(
            x_axis + offset,
            reduced_df.loc[(g,), 'mean'],
            width,
            yerr=reduced_df.loc[(g,), 'std'],
            label=g,
            capsize=width*15,
            ecolor='black',
            alpha=0.5)
        if bar_label:
            ax.bar_label(rect, padding=3)

    ax.set_xticks(x_axis, x_axis_labels)
    if ylim is not None:
        ax.set_ylim(ylim)

    if legend_title is None:
        legend_title = reduced_df.index.names[0]
    ax.legend(title=legend_title, fontsize=legend_fontsize)

    if x_label is None:
        x_label = reduced_df.index.names[1]
    ax.set_xlabel(x_label)

    if y_label is None:
        y_label = 'performance'
    ax.set_ylabel(y_label)

    plt.title(fig_title)

    if hline:
        plt.hlines(
            chance_level, np.min(x_axis) - 0.35, np.max(x_axis) + 0.35,
            label='Chance Level',
            linestyle='dotted', color='r',
            linewidth=2
        )

    adjust_figure()
    os.makedirs(osp.join(FIG_DIR, exp_name), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, exp_name, fig_name + '.pdf'), transparent=True)
    plt.close()


def bar_1par(config_df,
             exp_name,
             fig_name,
             chance_level=0.5,
             x_label=None,
             y_label=None,
             fig_title=None,
             perf_key='TrainAcc',
             ylim=[0, 1.5],
             hline=True,
             bar_label=True):
    """
        make a bar plot of the performance from a config_df
        that show one varying parameters and different random seeds.
        the variable parameters are different bars on the x axis

        args:
            config_df: a dataframe that only vary in the first two dimensions and random seeds
                with columns ['var1', 'var2', 'seed', 'config']
    """
    assert len(config_df.columns) == 4 \
           and config_df.columns[2] == 'seed' \
           and config_df.columns[3] == 'config'

    # get performance from log
    performance_list = [get_performance(cfg, perf_key) for cfg in config_df.loc[:, 'config']]
    config_df['performance'] = performance_list

    # group by the first 2 columns, this average performance across seeds
    groupby_obj = config_df.groupby(list(config_df.columns)[:2])
    reduced_df = groupby_obj.mean()
    reduced_df['mean'] = reduced_df['performance']
    reduced_df['std'] = groupby_obj.std(ddof=0)['performance']

    width = 0.7  # the width of the bars

    x_axis_labels = list(reduced_df.index)
    x_axis = np.arange(len(x_axis_labels))  # the label locations

    fig, ax = plt.subplots()
    rect = ax.bar(
        x_axis,
        reduced_df.loc[:, 'mean'],
        width,
        yerr=reduced_df.loc[:, 'std'],
        capsize=width*15,
        ecolor='black',
        alpha=0.5)
    if bar_label:
        ax.bar_label(rect, padding=3)

    ax.set_xticks(x_axis, x_axis_labels, rotation=30)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(x_label)

    if y_label is None:
        y_label = 'performance'
    ax.set_ylabel(y_label)

    plt.title(fig_title)

    if hline:
        plt.hlines(
            chance_level, np.min(x_axis) - 0.35, np.max(x_axis) + 0.35,
            label='Chance Level',
            linestyle='dotted', color='r',
            linewidth=2
        )

    adjust_figure()
    os.makedirs(osp.join(FIG_DIR, exp_name), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, exp_name, fig_name + '.pdf'), transparent=True)
    plt.close()
