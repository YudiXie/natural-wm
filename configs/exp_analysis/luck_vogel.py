import os.path as osp
from matplotlib import cm
import matplotlib.colors as mcolors

import numpy as np
import analysis.plots as plots
from analysis.plots import make_rgb_with_alpha
import configs.experiments as experiments
from utils.config_utils import configs_df2config_dict
from configs.exp_analysis.utils import *

__all__ = [
    'luckvogel_ngym_analysis',
    'luckvogel_decoding_analysis',
    'luckvogel_embedding_distance_analysis',
    'luckvogel_model_size_analysis',
    'luckvogel_change_magnitude_analysis',
    'luckvogel_ROC_analysis',
    'luckvogel_diff_cnn_analysis',
    'luckvogel_compare_att_analysis',
    'luckvogel_noatt_analysis',
    # The following are unused
    'luck_vogel_compare_norm_analysis',
    'luck_vogel_cnnsize_analysis',
    'luck_vogel_rnnsize_analysis',
    'luck_vogel_nonpret_modelsize_analysis',
    'luck_vogel_pret_analysis',
    'luck_vogel_compare_att_layers_analysis',
    'luck_vogel_delay_step_analysis',
    'luck_vogel_sample_step_analysis',
]

def luckvogel_noatt_analysis():
    cfgs = experiments.luckvogel_noatt()

    model_list = ['End-to-end', 'Pre-trained']
    set_sizes = [1, 2, 3, 4, 8, 12, ]
    performance = {}
    learning_curves = []

    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)
        learning_curves.append(get_learning_curve(cfgs, i))

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        model_list,
        f'AccvsSetSize_pret_vs_nonpret_noatt',
        colors=['#5AA9E6', '#FF6392'],
        label_list=['End-to-end', 'Pre-trained'],
        xticks=(4, 8, 12)
    )

def luckvogel_ngym_analysis():
    cfgs = experiments.luckvogel_ngym()

    model_list = ['End-to-end', 'Pre-trained']
    set_sizes = [1, 2, 3, 4, 8, 12, ]
    performance = {}
    learning_curves = []

    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)
        learning_curves.append(get_learning_curve(cfgs, i))

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        model_list,
        f'AccvsSetSize_pret_vs_nonpret',
        colors=['#5AA9E6', '#FF6392'],
        label_list=['End-to-end', 'Pre-trained'],
        xticks=(4, 8, 12)
    )

    cfg = cfgs.loc[0, 'config']
    x_axis = np.arange(cfg.log_every, cfg.max_batch + 1, cfg.log_every)

    def vertical_line(plt, x_axis, linewidth, capsize, capthick):
        # draw a dashed vertical line at x = 10000
        plt.vlines(
            x=10000,
            ymin=0,
            ymax=1.2,
            colors='gray',
            linestyles='dashed',
            linewidth=linewidth * 0.6
        )

    plots.error_plot(
        x_axis,
        learning_curves, 
        'Training Step',
        'Test Loss',
        fig_name=f'LearningCurve_pret_vs_nonpret',
        fig_dir='luck_vogel',
        colors=['#5AA9E6', '#FF6392'],
        label_list=model_list,
        mode='errorshade',
        xticks=[0, 10000, cfg.max_batch],
        extra_lines=vertical_line,
    )

def luckvogel_model_size_analysis():
    cfgs = experiments.luckvogel_model_size()
    set_sizes = [1, 2, 3, 4, 8, 12, ]
    performance = {}
    
    # compare pre-training to end-to-end training
    label_list = [
        'End-to-end (64, 128)',
        'Class CIFAR-10 pre-trained (64, 128)',
        'Contra CIFAR-10 pre-trained (64, 128)',
    ]
    exp_idx = [0, 5, 11]
    color1 = mcolors.to_rgb('#5AA9E6')  # blue for end-to-end trained
    color_alpha1 = [1.0]
    color2 = mcolors.to_rgb('#FF6392')  # red for pre-trained
    color_alpha2 = [1.0, 0.6]
    colors = [make_rgb_with_alpha(color1, a) for a in color_alpha1] + \
             [make_rgb_with_alpha(color2, a) for a in color_alpha2]

    for label, i in zip(label_list, exp_idx):
        performance[label] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        label_list,
        'pret_vs_end2end',
        legend_fontsize=5,
        chance_level_label=None,
        colors=colors,
        legend_loc=(0.01, 0.1),
    )

    # compare end-to-end training with different model sizes
    label_list = [
        'End-to-end (64, 128)',
        'End-to-end (64, 64)',
        'End-to-end (64, 32)',
        'End-to-end (32, 128)',
        'End-to-end (16, 128)',
    ]
    exp_idx = [0, 1, 2, 3, 4]
    y_offsets = [0.0, -0.01, -0.02, -0.03, -0.04]
    color1 = mcolors.to_rgb('#5AA9E6')  # blue for end-to-end trained
    color_alpha1 = [1.0, 0.8, 0.6, 0.4, 0.2]
    colors = [make_rgb_with_alpha(color1, a) for a in color_alpha1]

    for label, i in zip(label_list, exp_idx):
        performance[label] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        label_list,
        'end2end_model_size', 
        legend_fontsize=5,
        chance_level_label=None,
        colors=colors,
        y_offsets=y_offsets,
    )

    # compare classification pretraining with different model sizes
    label_list = [
        'Class CIFAR-10 pre-trained (64, 128)',
        'Class CIFAR-10 pre-trained (64, 64)',
        'Class CIFAR-10 pre-trained (64, 32)',
        'Class CIFAR-10 pre-trained (32, 128)',
        'Class CIFAR-10 pre-trained (16, 128)',
    ]
    exp_idx = [5, 6, 7, 8, 9]
    color1 = mcolors.to_rgb('#FF6392')  # blue for end-to-end trained
    color_alpha1 = [1.0, 0.8, 0.6, 0.4, 0.2]
    colors = [make_rgb_with_alpha(color1, a) for a in color_alpha1]
    for label, i in zip(label_list, exp_idx):
        performance[label] = get_luck_vogel_performance(cfgs, i)
    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        label_list,
        'class_pret_model_size', 
        legend_fontsize=5,
        chance_level_label=None,
        colors=colors,
        y_offsets=y_offsets,
    )

    # compare different pre-training methods
    label_list = [
        'Clas. CIFAR-10 pre-trained (64, 128)',
        'Clas. MNIST pre-trained (64, 128)',
        'Contra. pre-trained (64, 128)',
        'Random CNN (64, 128)',
    ]
    exp_idx = [5, 10, 11, 12]

    color2 = mcolors.to_rgb('#FF6392')  # red for pre-trained
    color_alpha2 = [1.0, 0.1, 0.6]
    colors = [make_rgb_with_alpha(color2, a) for a in color_alpha2]
    colors[1] = mcolors.to_rgb('green')
    colors = colors + [mcolors.to_rgb('k')]

    for label, i in zip(label_list, exp_idx):
        performance[label] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        label_list,
        'compare_pret_model_other', 
        legend_fontsize=5,
        chance_level_label=None,
        colors=colors,
        y_offsets=y_offsets,
    )

def luckvogel_diff_cnn_analysis():
    cfgs = experiments.luckvogel_diff_cnn()
    set_sizes = [1, 2, 3, 4, 8, 12, ]
    performance = {}

    label_list = [
        'AlexNet',
        'ResNet-18',
        'ResNet-50',
        'ViT'
    ]
    exp_idx = [5, 6, 7, 8]
    for label, i in zip(label_list, exp_idx):
        performance[label] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        label_list,
        'diff_cnn_architectures', 
        legend_fontsize=10,
        chance_level_label=None,
    )

    label_list = ['Random ' + label for label in label_list]
    exp_idx = [9, 10, 11, 12]
    for label, i in zip(label_list, exp_idx):
        performance[label] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        label_list,
        'diff_random_cnn_architectures', 
        legend_fontsize=10,
        chance_level_label=None,
    )

    exp_idx = [1, 2, 7, 3, 4]
    label_list = ['R3M ResNet', 'VC-1 ViT', 'Classification ResNet', 'Autoencoding ResNet', 'Inpainting ResNet']
    for label, i in zip(label_list, exp_idx):
        performance[label] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        label_list,
        'diff_cnn_pret', 
        legend_fontsize=10,
        chance_level_label=None,
    )

def luckvogel_ROC_analysis():
    from analysis.luck_vogel_roc import get_hit_and_false_alarm_rates, get_scores_and_targets

    cfgs = experiments.luckvogel_ngym()
    num_seeds = 4
    set_sizes = (1, 2, 3, 4, 8, 12)
    fig_dir = 'luckvogel_ROC'
    
    # load targets and scores, 
    # both are lists of lists (num_seeds, num_set_sizes, num_trials)
    scores_list = [[] for _ in range(num_seeds)]
    targets_list = [[] for _ in range(num_seeds)]
    for seed in range(num_seeds):
        cfg = cfgs.loc[seed * 2 + 1, 'config']
        scores_list[seed], targets_list[seed] = get_scores_and_targets(cfg)

    # compute how hit rate and false alarm rate change with set size
    # when threshold is 0.5
    hit_rates = [[] for _ in set_sizes]
    false_alarm_rates = [[] for _ in set_sizes]
    for seed in range(num_seeds):
        for i_ss in range(len(set_sizes)):
            scores = np.array(scores_list[seed][i_ss])
            targets = np.array(targets_list[seed][i_ss])
            hr, far = get_hit_and_false_alarm_rates(scores, targets, 0.5)
            hit_rates[i_ss].append(hr)
            false_alarm_rates[i_ss].append(far)
    
    plots.error_plot(
        x_axis=set_sizes,
        data_list=[hit_rates, false_alarm_rates],
        x_label='Set size',
        y_label='Rate',
        label_list=['Hit rate', 'False alarm rate'],
        fig_dir=fig_dir,
        fig_name='hit_false_alarm_rate_threshold_0_5',
        mode='errorbar2',
        legend_fontsize=11,
        colors=['seagreen', 'darkorchid'],
        yticks=[0.0, 0.5, 1.0],
    )

    plot_idx = [1, 3, 4, 5]

    # plot ROC curves of different set sizes, 
    # pool data from all random seeds together
    # each of the lists are of size (num_set_sizes, num_trials)
    scores_list_pool_seeds = []
    targets_list_pool_seeds = []
    for i_ss in plot_idx:
        scores_list_pool_seeds.append([s for seed in range(num_seeds) 
                                       for s in scores_list[seed][i_ss]])
        targets_list_pool_seeds.append([t for seed in range(num_seeds) 
                                        for t in targets_list[seed][i_ss]])
    plots.plot_ROC_curves(scores_list_pool_seeds,
                          targets_list_pool_seeds,
                          [set_sizes[idx] for idx in plot_idx],
                          fig_dir=fig_dir)

def luckvogel_decoding_analysis():
    from analysis.luck_vogel_decoding import \
        luck_vogel_decoding, compare_decoder_model_behavior, \
        get_luck_vogel_decoder_performance, get_luck_vogel_similarity_scores
    
    cfgs = experiments.luckvogel_ngym()
    layers = ['resblock1', 'resblock2', 'resblock3', 'cnn', ]
    layer_names = ['ResBlock1', 'ResBlock2', 'ResBlock3', 'CNN output', ]
    model_list = ['ResNet', 'pretResNet', ]
    set_sizes = [1, 2, 3, 4, 8, 12, ]

    # The following loop can be skipped if the results are already saved
    for i in cfgs.index:
        luck_vogel_decoding(cfgs.loc[i, 'config'], record_layers=layers, test_batch=50, n_iter=1)
        compare_decoder_model_behavior(cfgs.loc[i, 'config'], record_layers=layers, test_batch=10, layer_names=layer_names)

    performance = {}
    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)
    decoder_performance = {}
    for i, model in enumerate(model_list):
        decoder_performance[model] = {}
        for layer in layers:
            decoder_performance[model][layer] = get_luck_vogel_decoder_performance(cfgs, i, layer)

    for i, model in enumerate(model_list):
        decoder_performance[model]['model'] = performance[model]
        plots.luckvogel_errorbar_plot(
            decoder_performance[model], 
            set_sizes, 
            layers + ['model', ],
            f'luckvogel_decoding_{model}', 
            xticks=(4, 8, 12),
            label_list=layer_names + ['CNN+RNN Performance'],
            colors=['#00b4d8', '#0096c7', '#0077b6', '#023e8a', '#031a62', 'black'],
            chance_level_label=None,
            legend_fontsize=11,
            plot_dir='luckvogel_decoding'
        )

        for layer, layer_name in zip(layers, layer_names):
            similarity, baseline = get_luck_vogel_similarity_scores(cfgs, i, layer)
            plots.error_plot(
                set_sizes,
                [similarity, baseline],
                x_label='Set Size',
                y_label='Behavioral Similarity',
                fig_dir='luckvogel_decoding',
                fig_name=f'{model}_{layer}_behavioral_similarity_vs_set_size',
                colors=['lightblue', 'gray'],
                label_list=['Decoder', 'Baseline'],
                ylim=[0.5, 1.05],
                yticks=[0.6, 0.8, 1.0],
                xticks=(4, 8, 12),
            )

def luckvogel_embedding_distance_analysis():
    from analysis.luck_vogel_decoding import compute_embedding_distances, get_luck_vogel_embedding_distances
    
    cfgs = experiments.luckvogel_ngym()
    layers = ['resblock1', 'resblock2', 'resblock3', 'cnn', ]
    layer_names = ['ResBlock1', 'ResBlock2', 'ResBlock3', 'CNN output', ]
    model_list = ['ResNet', 'pretResNet', ]
    set_sizes = [1, 2, 3, 4, 8, 12, ]

    # The following loop can be skipped if the results are already saved
    for i in cfgs.index:
        compute_embedding_distances(cfgs.loc[i, 'config'], record_layers=layers, test_batch=10)

    performance = {}
    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)

    keys = ['signal', 'noise', 'snr', 'log_snr']
    y_labels = ['Signal', 'Noise', 'SNR', 'log(SNR)']
    
    embedding_distance = {}
    for i, model in enumerate(model_list):
        embedding_distance[model] = {}
        for layer in layers:
            embedding_distance[model][layer] = get_luck_vogel_embedding_distances(cfgs, i, layer)

    for key, y_label in zip(keys, y_labels):
        for model in model_list:
            for layer, layer_name in zip(layers, layer_names):
                plots.error_plot(
                    set_sizes,
                    [embedding_distance[model][layer][key]],
                    x_label='Set Size',
                    y_label=y_label,
                    ylim=[0.0, None],
                    fig_dir='luckvogel_embed_distance',
                    fig_name=f'{key}_vs_set_size_{layer}_{model}',
                    xticks=(4, 8, 12),
                    colors=['#5AA9E6' if model == 'ResNet' else '#FF6392'],
                    label_list=['End-to-end' if model == 'ResNet' else 'Pre-trained'],
                )

def luck_vogel_compare_norm_analysis():
    cfgs = experiments.luck_vogel_compare_norm()

    set_sizes = [1, 2, 3, 4, 8, 12]
    model_list = ['ResNet', 'pretResNet', ]

    for k, norm in enumerate(['groupnorm', 'instancenorm', 'batchnorm', 'layernorm', ]):
        for j, att in enumerate(['film', 'cbam', ]):

            performance = {}
            for i, model in enumerate(model_list):
                performance[model] = get_luck_vogel_performance(cfgs, (k * 2 + j) * 2 + i)

            plots.luckvogel_errorbar_plot(
                performance, set_sizes, 
                model_list,
                f'comparenorm_{att}_{norm}',
                xticks=(4, 8, 12)
            )

def luck_vogel_cnnsize_analysis():
    cfgs = experiments.luck_vogel_cnnsize()

    set_sizes = [1, 2, 3, 4, 8, 12]
    model_list = [16, 32, 64, 128, 256, ]

    performance = {}
    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, set_sizes, 
        model_list,
        f'cnnsize',
        legend_title='CNN size'
    )
    
    indices = [3, 4, 5]
    capacity = [[(np.array(performance[model][i]) - 0.5) * set_sizes[i] * 2 for model in model_list] for i in indices]
    set_sizes = [4, 8, 12]

    plots.error_plot(
        np.log2(model_list),
        capacity,
        'log2(CNN size)',
        y_label='Item Capacity',
        label_list=set_sizes,
        fig_dir='luck_vogel_capacity',
        fig_name='logcnn_cap',
        legend_title='Set size'
    )

    plots.error_plot(
        model_list,
        capacity,
        'CNN size',
        y_label='Item Capacity',
        label_list=set_sizes,
        fig_dir='luck_vogel_capacity',
        fig_name='cnn_cap',
        legend_title='Set size'
    )

def luck_vogel_rnnsize_analysis():
    cfgs = experiments.luck_vogel_rnnsize()

    set_sizes = [1, 2, 3, 4, 8, 12]
    model_list = [32, 64, 128, 256, 512, 1024]

    performance = {}
    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        model_list,
        f'rnnsize',
        legend_title='RNN size'
    )
    
    indices = [3, 4, 5]
    capacity = [[(np.array(performance[model][i]) - 0.5) * set_sizes[i] * 2 for model in model_list] for i in indices]
    set_sizes = [4, 8, 12]

    plots.error_plot(
        np.log2(model_list),
        capacity,
        'log2(RNN size)',
        y_label='Item Capacity',
        label_list=set_sizes,
        fig_dir='luck_vogel_capacity',
        fig_name='logrnn_cap',
        legend_title='Set size'
    )

    plots.error_plot(
        model_list,
        capacity,
        'RNN size',
        y_label='Item Capacity',
        label_list=set_sizes,
        fig_dir='luck_vogel_capacity',
        fig_name='rnn_cap',
        legend_title='Set size'
    )

def luck_vogel_nonpret_modelsize_analysis():
    cfgs = experiments.luck_vogel_nonpret_modelsize()

    model_list = [1, 2, 4]
    performance = {}

    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        [1, 2, 3, 4, 8, 12],
        [1, 4],
        f'nonpret_compare_rnnsize',
        legend_title='RNN size',
        chance_level_label=None
    )

    """
    plots.luckvogel_errorbar_plot(
        performance, 
        [1, 2, 3, 4, 8, 12],
        [4, 16],
        f'nonpret_compare_modelsize_selected',
        legend_title='Model size',
        chance_level_label=None
    )
    """

def luck_vogel_pret_analysis():
    cfgs = experiments.luck_vogel_pret()

    for k, freeze in enumerate(['frozen', 'nonfrozen']):

        model_list = ['LV count', 'LV simclr', 'CIFAR classification', 'CIFAR simclr']
        performance = {}

        for i, model in enumerate(model_list):
            performance[model] = get_luck_vogel_performance(cfgs, i * 2 + k)

        plots.luckvogel_errorbar_plot(
            performance, 
            [1, 2, 3, 4, 8, 12], 
            model_list[-2: ],
            f'compare_pretraining_{freeze}cnn_dataset_cifar'
        )

        plots.luckvogel_errorbar_plot(
            performance, 
            [1, 2, 3, 4, 8, 12], 
            model_list[: 2],
            f'compare_pretraining_{freeze}cnn_dataset_luckvogel'
        )

def luckvogel_compare_att_analysis():
    cfgs = experiments.luckvogel_compare_att()

    model_list = ['No attention', 'CBAM', 'FiLM', 'elementwise', ]
    set_sizes = [1, 2, 3, 4, 8, 12]
    performance = {}

    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        model_list,
        'AccvsSetSize_compare_attention',
        chance_level_label=None,
        legend_fontsize=10,
        colors=['#C3E0E5', '#41729F', '#5885AF', '#274472']
    )

def luck_vogel_compare_att_layers_analysis():
    cfgs = experiments.luck_vogel_compare_att_layers()

    model_list = ['First', 'Second', 'Third', 'All']
    set_sizes = cfgs[0][0].num_patches
    performance = {}

    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        model_list,
        'luckvogel_att_layers', 2
    )

def luck_vogel_delay_step_analysis():
    cfgs = experiments.luck_vogel_delay_step()

    model_list = [10, 30, 50, 70]
    set_sizes = [1, 2, 3, 4, 8, 12]
    performance = {}

    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        model_list,
        'compare_delay_step',
        chance_level_label=None,
        legend_title='Delay Step'
    )

    accuracy = []
    plot_set_sizes = [4, 8, 12]
    for j, size in enumerate(set_sizes):
        if size in plot_set_sizes:
            accuracy.append([])
            for model in model_list:
                accuracy[-1].append(performance[model][j])

    plots.error_plot(
        model_list,
        accuracy,
        "Delay Step",
        "Accuracy",
        plot_set_sizes,
        fig_dir='luck_vogel',
        fig_name='compare_delay_step_',
        legend_title='Set Size'
    )

def luck_vogel_sample_step_analysis():
    cfgs = experiments.luck_vogel_sample_step_test()

    model_list = [1, 2, 4, 8, 16]
    set_sizes = cfgs[0][0].num_patches
    performance = {}

    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        model_list,
        'luckvogel_sample_step',
        legend_title='Sample Step'
    )

    accuracy = []
    plot_set_sizes = [4, 8, 12]
    for j, size in enumerate(set_sizes):
        if size in plot_set_sizes:
            accuracy.append([])
            for model in model_list:
                accuracy[-1].append(performance[model][j])

    plots.error_plot(
        model_list,
        accuracy,
        "Sample Step",
        "Accuracy",
        plot_set_sizes,
        fig_dir='luck_vogel',
        fig_name='luckvogel_sample_step_',
        legend_title='Set Size'
    )

def luckvogel_change_magnitude_analysis():
    cfgs = experiments.luckvogel_change_magnitude_test()

    model_list = list(range(0, 100, 10))
    set_sizes = cfgs.loc[0].config.num_patches
    performance = {}
    hit_rates = {}

    for j, n in enumerate(set_sizes):
        hit_rates[n] = []

    for i, model in enumerate(model_list):
        performance[model] = get_luck_vogel_performance(cfgs, i)
        ret = get_luck_vogel_performance(cfgs, i, key='hit_rate')

        for j, n in enumerate(set_sizes):
            hit_rates[n].append(ret[j])

    plots.luckvogel_errorbar_plot(
        performance, 
        set_sizes, 
        [10, 30, 90],
        'compare_change_magnitude',
        legend_title='Change Magnitude',
        chance_level_label=None
    )

    plot_set_sizes = [1, 2, 4, 8]
    cmap = cm.get_cmap('cool')
    colors = [cmap(1 - i) for i in np.linspace(0.0, 1.0, len(plot_set_sizes))]

    plots.error_plot(
        model_list,
        [hit_rates[n] for n in plot_set_sizes],
        'Magnitude of change (deg)',
        'Proportion report "change"',
        plot_set_sizes,
        fig_dir='luck_vogel',
        legend_title='Set size',
        fig_name='change_magnitude_hit_rate',
        xticks=[0, 30, 60, 90],
        yticks=[0, 0.4, 0.8],
        legend_fontsize=12,
        fontsize=12,
        mode='errorshade',
        colors=colors,
    )