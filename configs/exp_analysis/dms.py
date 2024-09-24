import torch
import numpy as np
import os.path as osp
import pandas as pd
import analysis.plots as plots
import configs.experiments as experiments

from .luck_vogel import get_luck_vogel_performance

__all__ = [
    'dms_compare_modalities_analysis',
    'dms_cross_modality_gen_analysis',
    'dms_luck_vogel_gen_analysis',
    'dms_luck_vogel_gen_train_linear_analysis',
    'dms_luck_vogel_gen_cnnsize_analysis',
    'dms_luck_vogel_gen_train_linear_cnnsize_analysis',
    'dms_cross_dataset_gen_analysis',
    'dms_cross_dataset_gen_distortion_analysis',
    'dms_compare_datasets_analysis',
    'dms_compare_datasets_distortion_analysis',
    'dms_variable_sample_time_att_analysis'
]

def dms_compare_modalities_analysis():
    cfgs = experiments.dms_compare_modalities()

    dataset_list = [
        ('Speech', 'GLOVE'),
        ('GLOVE', 'CIFAR10'),
        ('Speech', 'CIFAR10'),
        ('Speech', 'GLOVE', 'CIFAR10'),
        'SpeechCommands', 'GLOVE', 'CIFAR10'
    ]
    performance = {}

    eval_interval = cfgs[0][0].log_every
    tot_steps = 50
    plot_every = 5

    for i, dataset in enumerate(dataset_list):
        performance[dataset] = []
        for seed in range(4):
            cfg = cfgs[seed][i]
            try:
                file_path = osp.join(cfg.save_path, 'progress.txt')
                exp_data = pd.read_table(file_path)
                acc = exp_data['TestAcc'][plot_every - 1: tot_steps: plot_every]
                if len(acc) == tot_steps // plot_every:
                    acc = np.concatenate([np.array([50, ], ), acc], axis=0)
                    performance[dataset].append(acc)
                else:
                    raise RuntimeError
            except:
                print("Incomplete data:", cfg.save_path)

    x_axis = np.arange(0, tot_steps * eval_interval + 1, plot_every * eval_interval)

    plots.dms_learning_curves_plot(
        performance,
        dataset_list[:4],
        x_axis,
        'multimodal_curves_joint'
    )

    plots.dms_learning_curves_plot(
        performance,
        dataset_list[-3:],
        x_axis,
        'multimodal_curves_single'
    )

def dms_cross_modality_gen_analysis():
    cfgs = experiments.dms_cross_modality_gen()[0]

    dataset_list =  ['SpeechCommands', 'GLOVE', 'CIFAR10', ]
    train_dataset_list = [
        ('Speech', 'GLOVE'),
        ('GLOVE', 'CIFAR10'),
        ('Speech', 'CIFAR10'),
        'All',
        'Speech', 'GLOVE', 'CIFAR10'
    ]
    m = len(dataset_list)
    performance = np.zeros((7, 3))

    for k, eval_dataset in enumerate(dataset_list):
        for i, train_dataset in enumerate(train_dataset_list):
            
            for seed in range(4):
                cfg = cfgs[(seed * len(train_dataset_list) + i) * m + k]
                # print(model, eval_dataset, train_dataset, cfg.load_path, cfg.dataset)
                try:
                    file_path = osp.join(cfg.save_path, 'progress.txt')
                    exp_data = pd.read_table(file_path)
                    acc = exp_data['TestAcc'][0]
                    performance[i, k] += acc
                except:
                    print("Incomplete data:", cfg.save_path)

    performance /= 4
    plots.dms_cross_dataset_gen_plot(
        performance[:4], 
        dataset_list, 
        f"cross_modality_gen_joint",
        train_dataset_list[:4])

    plots.dms_cross_dataset_gen_plot(
        performance[-3:], 
        dataset_list, 
        f"cross_modality_gen_single",
        train_dataset_list[-3:])

def dms_luck_vogel_gen_analysis():
    cfgs = experiments.dms_luck_vogel_gen()

    dataset_list = ['Omniglot-Colored', 'MNIST-Colored', 'CIFAR10', 'CIFAR100', 'WhiteNoise']
    model_list = ['pretResNet', 'ResNet']
    
    n = len(model_list)
    m = len(dataset_list)
    num_patches = cfgs[0][0].num_patches
    num_seeds = 4
    
    for j, model in enumerate(model_list):

        performance = {}
        for i, train_dataset in enumerate(dataset_list):
            for seed in range(num_seeds):
                performance[train_dataset] = get_luck_vogel_performance(cfgs, i * n + j)

        plots.luck_vogel_plot_errorbar(
            performance,
            num_patches,
            dataset_list,
            f'dms_gen_{model}'
        )

def dms_luck_vogel_gen_train_linear_analysis():
    cfgs = experiments.dms_luck_vogel_gen_train_linear()

    dataset_list = ['Omniglot-Colored', 'MNIST-Colored', 'CIFAR10', 'CIFAR100', 'WhiteNoise']
    model_list = ['pretResNet', 'ResNet']
    
    n = len(model_list)
    m = len(dataset_list)
    num_patches = cfgs[0][0].num_patches
    num_seeds = 4
    
    for j, model in enumerate(model_list):

        performance = {}
        for i, train_dataset in enumerate(dataset_list):
            for seed in range(num_seeds):
                performance[train_dataset] = get_luck_vogel_performance(cfgs, i * n + j)

        plots.luck_vogel_plot_errorbar(
            performance,
            num_patches,
            dataset_list,
            f'dms_gen_linear_{model}'
        )

def dms_luck_vogel_gen_cnnsize_analysis():
    cfgs = experiments.dms_luck_vogel_gen_cnnsize()

    size_list = [16, 32, 64, 128, 256]
    model_list = ['pretResNet', 'ResNet']
    
    n = len(model_list)
    m = len(size_list)
    num_patches = cfgs[0][0].num_patches
    num_seeds = 4
    
    for i, model in enumerate(model_list):

        performance = {}
        for j, size in enumerate(size_list):
            for seed in range(num_seeds):
                performance[size] = get_luck_vogel_performance(cfgs, j * n + i)

        plots.luck_vogel_plot_errorbar(
            performance,
            num_patches,
            size_list,
            f'dms_gen_cnnsize_{model}'
        )

def dms_luck_vogel_gen_train_linear_cnnsize_analysis():
    cfgs = experiments.dms_luck_vogel_gen_train_linear_cnnsize()

    size_list = [16, 32, 64, 128, 256]
    model_list = ['pretResNet', 'ResNet']
    
    n = len(model_list)
    m = len(size_list)
    num_patches = cfgs[0][0].num_patches
    num_seeds = 4
    
    for i, model in enumerate(model_list):

        performance = {}
        for j, size in enumerate(size_list):
            for seed in range(num_seeds):
                performance[size] = get_luck_vogel_performance(cfgs, j * n + i)

        plots.luck_vogel_plot_errorbar(
            performance,
            num_patches,
            size_list,
            f'dms_gen_linear_cnnsize_{model}'
        )

def dms_cross_dataset_gen_analysis(distortion=False):
    cfgs = experiments.dms_cross_dataset_gen(distortion)[0]
    # seed * train_datasets * model * eval_datasets

    dataset_list = ['Omniglot-Colored', 'MNIST-Colored', 'CIFAR10', 'CIFAR100', 'WhiteNoise']
    model_list = ['pretResNetcbamCTRNN', 'pretResNetCTRNN', 'ResNetcbamCTRNN', 'ResNetCTRNN']
    
    n = len(model_list)
    m = len(dataset_list)
    
    performances = {}
    for j, model in enumerate(model_list):

        performance = np.zeros((5, 5))
        performances[model] = np.zeros((4, 5, 5))

        for k, eval_dataset in enumerate(dataset_list):
            for i, train_dataset in enumerate(dataset_list):
                
                for seed in range(4):
                    cfg = cfgs[((seed * m + i) * n + j) * m + k]
                    # print(model, eval_dataset, train_dataset, cfg.load_path, cfg.dataset)
                    try:
                        file_path = osp.join(cfg.save_path, 'progress.txt')
                        exp_data = pd.read_table(file_path)
                        acc = exp_data['TestAcc'][0]
                        performance[i, k] += acc
                        performances[model][seed, i, k] = acc
                    except:
                        print("Incomplete data:", cfg.save_path)

        performance /= 4
        plots.dms_cross_dataset_gen_plot(
            performance, 
            dataset_list, 
            f"{'dist_' if distortion else ''}{model}_cross_dataset_gen")

    print(performances)

    plots.dms_compare_generalization_plot(
        performances,
        ['pretResNetcbamCTRNN', 'ResNetcbamCTRNN'],
        dataset_list,
        f"{'dist_' if distortion else ''}compare_pret_nonpret_gen_cbam"
    )

    plots.dms_compare_generalization_plot(
        performances,
        ['pretResNetCTRNN', 'ResNetCTRNN'],
        dataset_list,
        f"{'dist_' if distortion else ''}compare_pret_nonpret_gen_noatt"
    )

def dms_cross_dataset_gen_distortion_analysis():
    dms_cross_dataset_gen_analysis(True)

def dms_compare_datasets_analysis(distortion=False):
    cfgs = experiments.dms_compare_datasets()
    if distortion:
        cfgs = experiments.dms_compare_datasets_distortion()

    dataset_list = ['Omniglot-Colored', 'MNIST-Colored', 'CIFAR10', 'CIFAR100', 'WhiteNoise']
    model_list = ['pretResNet', 'ResNet']
    performance = {}

    eval_interval = cfgs[0][0].log_every
    tot_steps = 40
    plot_every = 2

    for j, model in enumerate(model_list):
        for i, dataset in enumerate(dataset_list):
            performance[dataset] = []
            for seed in range(4):
                cfg = cfgs[seed][i * len(model_list) + j]
                try:
                    file_path = osp.join(cfg.save_path, 'progress.txt')
                    exp_data = pd.read_table(file_path)
                    acc = exp_data['TestAcc'][plot_every - 1: tot_steps: plot_every]
                    if len(acc) == tot_steps // plot_every:
                        acc = np.concatenate([np.array([50, ], ), acc], axis=0)
                        performance[dataset].append(acc)
                except:
                    print("Incomplete data:", cfg.save_path)

        x_axis = np.arange(0, tot_steps * eval_interval + 1, plot_every * eval_interval)

        plots.dms_learning_curves_plot(
            performance,
            dataset_list,
            x_axis,
            f'{model}_curves' if not distortion else f'{model}_distortion_curves'
        )
    
def dms_compare_datasets_distortion_analysis():
    dms_compare_datasets_analysis(True)

def dms_variable_sample_time_att_analysis():
    cfgs = experiments.dms_variable_sample_time_att()

    model_list = ['film_inst', 'cbam_inst', 'film_batch', 'cbam_batch']

    performance = {}
    for i, model in enumerate(model_list):
        performance[model] = [[] for _ in range(5)]
        for seed in range(2):
            configs = cfgs[seed][i * 5: i * 5 + 5]
            for j, cfg in enumerate(configs):
                try:
                    file_path = osp.join(cfg.save_path, 'progress.txt')
                    exp_data = pd.read_table(file_path)
                    acc = exp_data['TestAcc'][-10:].mean()
                    performance[model][j].append(acc)
                except:
                    print("Incomplete data:", cfg.save_path)

    plots.dms_variable_sample_time_plot(
        performance, model_list, 
        'dms_sampletime_pretrain_att',
        [1, 2, 4, 8, 16]
    )