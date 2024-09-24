import copy

import numpy as np
from matplotlib import cm

import analysis.plots as plots
import configs.experiments as experiments
from analysis.circular_stat import fit_vonmises_uniform, center_angle, \
    fit_von_mises_uniform_mixture, get_density_fit_and_residual
from configs.exp_analysis.utils import *

__all__ = [
    'continuous_report_cnn_size_analysis',
    'continuous_report_rnn_size_analysis',
    'continuous_report_cnn_size_end2end_analysis',
    'continuous_report_rnn_size_end2end_analysis',
    'continuous_report_analysis',
    'continuous_report_diff_cnn_analysis',
    'continuous_report_noatt_analysis',
    'continuous_report_nontarget_analysis',
    'continuous_report_compare_norm_analysis',
    'continuous_report_with_uncertainty_analysis',
    'cued_continuous_report_analysis',
    'cued_continuous_report_set_size_analysis',
    'continuous_report_vonmises_fit_analysis'
]

def continuous_report_cnn_size_analysis():
    continuous_report_model_size_analysis(
        experiments.continuous_report_cnn_size(), 0,
        [8, 16, 32, 64, 128, 256],
        'continuous_report_model_size',
        'pretrained',
        module='CNN',
        plot_model_size_idx=list(range(2, 6)),
        SD_vs_setsize_kwargs=dict(ylim=[0, 90]),
        SD_vs_modelsize_kwargs=dict(ylim=[0, 90]),
        precision_vs_modelsize_kwargs=dict(ylim=[0, 3.5])
    )

def continuous_report_rnn_size_analysis():
    continuous_report_model_size_analysis(
        experiments.continuous_report_rnn_size(), 0,
        [32, 64, 128, 256, 512, 1024],
        'continuous_report_model_size',
        'pretrained',
        plot_model_size_idx=list(range(2, 6)),
        SD_vs_modelsize_kwargs=dict(ylim=[0, 120]),
        precision_vs_modelsize_kwargs=dict(ylim=[0, 2])
    )

def continuous_report_rnn_size_end2end_analysis():
    continuous_report_model_size_analysis(
        experiments.continuous_report_rnn_size_end2end(), 0,
        [4, 8, 16, 32, ],
        'continuous_report_model_size',
        'endtoend',
        plot_model_size_idx=[0, 1, ],
        SD_vs_setsize_kwargs=dict(errormode='std'),
    )

def continuous_report_cnn_size_end2end_analysis():
    continuous_report_model_size_analysis(
        experiments.continuous_report_cnn_size_end2end(), 0,
        [4, 8, 16, 32, ],
        'continuous_report_model_size',
        'endtoend',
        module='CNN',
        plot_model_size_idx=[0, 1, ],
        SD_vs_setsize_kwargs=dict(ylim=[0, 90]),
    )

# analysis for continuous report task
# plot1: error sd vs set size for each model
# plot2: precision vs set size for each model
# plot3: error distribution at different set sizes for all models
# plot4: learning curve for each model
def continuous_report_analysis():
    cfgs = experiments.continuous_report_ngym()

    model_list = ['ResNet', 'pretResNet', ]
    set_sizes = [1, 2, 3, 4, 5, 6]
    plot_set_sizes = [1, 2, 4, 6]
    yticks_list = [[0.0, 0.05, 0.10], [0.0, 0.02, 0.04]]
    cmap = cm.get_cmap('coolwarm')
    dist_colors_list = [
        [cmap(i) for i in np.linspace(0.0, 0.4, len(plot_set_sizes))],
        [cmap(1 - i) for i in np.linspace(0.0, 0.4, len(plot_set_sizes))],
    ]

    performance = {}
    precision = {}
    learning_curves = []

    for i, model in enumerate(model_list):
        performance[model], errors, precision[model] = \
            get_continuous_report_performance_error(cfgs, i)
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
        )

        # plot Precision with fitted curve
        plots.continuous_report_plot(
            precision,
            set_sizes,
            [model, ],
            f'{model}_precision',
            y_label='Precision (1 / rad)',
            fit_curve=True,
            colors=[(1, 127 / 255, 14 / 255), ] if i > 0 else None,
        )

    # plot how error sd changes with set size for two models
    plots.continuous_report_plot(
        performance,
        set_sizes,
        model_list,
        'recallSDvsSetSize_pret_vs_nonpret',
        legend_loc='upper left',
        legend_fontsize=11,
        colors=['#5AA9E6', '#FF6392'],
        label_list=['End-to-end', 'Pre-trained']
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
        fig_dir='continuous_report',
        colors=['#5AA9E6', '#FF6392'],
        label_list=['End-to-end', 'Pre-trained'],
        mode='errorshade',
        xticks=[0, 30000, 60000],
    )

def continuous_report_diff_cnn_analysis():
    cfgs = experiments.continuous_report_diff_cnn()

    set_sizes = [1, 2, 3, 4, 5, 6 ]
    performance = {}
    precision = {}

    label_list = [
        'AlexNet',
        'ResNet-18',
        'ResNet-50',
        'ViT'
    ]
    exp_idx = [5, 6, 7, 8]
    for label, i in zip(label_list, exp_idx):
        performance[label], errors, precision[label] = get_continuous_report_performance_error(cfgs, i)

    plots.continuous_report_plot(
        performance, 
        set_sizes, 
        label_list,
        'diff_cnn_architectures', 
        legend_fontsize=10,
    )

    label_list = ['Random ' + label for label in label_list]
    exp_idx = [9, 10, 11, 12]
    for label, i in zip(label_list, exp_idx):
        performance[label], errors, precision[label] = get_continuous_report_performance_error(cfgs, i)

    plots.continuous_report_plot(
        performance, 
        set_sizes, 
        label_list,
        'diff_random_cnn_architectures', 
        legend_fontsize=10
    )

    exp_idx = [1, 2, 7, 3, 4]
    label_list = ['R3M ResNet', 'VC-1 ViT', 'Classification ResNet', 'Autoencoding ResNet', 'Inpainting ResNet']
    for label, i in zip(label_list, exp_idx):
        performance[label], errors, precision[label] = get_continuous_report_performance_error(cfgs, i)

    plots.continuous_report_plot(
        performance, 
        set_sizes, 
        label_list,
        'diff_cnn_pret', 
        legend_fontsize=10,
    )

def continuous_report_taskonomy_analysis():
    cfgs = experiments.continuous_report_taskonomy()
    tasks = ['autoencoding', 'inpainting', 'segment_semantic', 'class_object', 'class_scene']
    performance = {}
    precision = {}

    for task in tasks:
        performance[task], errors, precision[task] = get_continuous_report_performance_error(cfgs, tasks.index(task))
    
    plots.continuous_report_plot(
        performance,
        [1, 2, 3, 4, 5, 6],
        tasks,
        'continuous_report_taskonomy',
        legend_fontsize=10,
    )


def continuous_report_nontarget_analysis():
    cfgs = experiments.continuous_report_ngym()

    model_list = ['End-to-end', 'Pretrained']
    set_sizes = [1, 2, 3, 4, 5, 6]
    plot_set_sizes = [2, 4, 6, ]

    from analysis.continuous_report_distribution import continuous_report_nontarget_distribution
    #for i in range(len(cfgs)):
    #    continuous_report_nontarget_distribution(cfgs.iloc[i].config)

    for i, model in enumerate(model_list):
        errors = get_continuous_report_nontarget_error(cfgs, i)

        # draw a basline at 15 / 360
        def extra_lines(plt, x_axis, linewidth, capsize, capthick):
            plt.axhline(y=15 / 360, color='gray', linestyle='--', linewidth=linewidth)

        # plot error distribution at different set sizes
        plots.error_distribution_plot(
            [errors[n - 1] for n in plot_set_sizes],
            plot_set_sizes,
            -150, 150, 15,
            f'{model}_nontarget_dist',
            x_label='Distance to non-target (deg)',
            legend_title='Set Size',
            coef=15,
            yticks=[0.03, 0.05, 0.07],
            xticks=[-90, 0, 90],
            extra_lines=extra_lines,
        )

def continuous_report_noatt_analysis():
    cfgs = experiments.continuous_report_noatt()

    model_list = ['ResNet', 'pretResNet', ]
    set_sizes = [1, 2, 3, 4, 5, 6]

    performance = {}
    precision = {}

    for i, model in enumerate(model_list):
        performance[model], errors, precision[model] = \
            get_continuous_report_performance_error(cfgs, i)

    # plot how error sd changes with set size for two models
    plots.continuous_report_plot(
        performance,
        set_sizes,
        model_list,
        'recallSDvsSetSize_pret_vs_nonpret_no_attention',
        legend_loc='upper left',
        legend_fontsize=11,
        colors=['#5AA9E6', '#FF6392'],
        label_list=['End-to-end', 'Pre-trained']
    )

def continuous_report_vonmises_fit_analysis():
    cfgs = experiments.continuous_report_ngym()

    model_list = ['ResNet', 'pretResNet', ]
    set_sizes = [1, 2, 3, 4, 5, 6]

    performance = {}
    precision = {}

    left = -120
    right = 120
    interval = 8
    x_axis = np.arange(left, right + 0.01, interval)

    for i, model in enumerate(model_list):
        performance[model], errors_diff_ssizes, precision[model] = \
            get_continuous_report_performance_error(cfgs, i)

        errors_all_ssizes_all_seeds = []
        w_diff_ssizes = []
        csd_diff_ssizes = []
        aggregate_residuals = []
        for i_ssize, errors_diff_seeds in enumerate(errors_diff_ssizes):
            errors_all_seeds = []
            w_diff_seeds = []
            csd_diff_seeds = []
            for seed, errors in enumerate(errors_diff_seeds):
                errors_all_seeds.extend(errors)

                # for each set size and each seed, fit mixture
                errors_centered = center_angle(copy.deepcopy(np.array(errors)) * np.pi / 180)
                w, kappa, csd = fit_von_mises_uniform_mixture(errors_centered)
                # plot histogram of errors, fit, and residuals
                density, mixture_fit, residuals = \
                    get_density_fit_and_residual(errors_centered, x_axis, w, kappa)
                plots.plot_density_fit_residual(x_axis, density, mixture_fit, residuals,
                                                title=f'Set Size: {set_sizes[i_ssize]}, Seed: {seed} \n' +
                                                      f'w: {w:.2f}, kappa: {kappa:.2f} \n' +
                                                      f'csd: {csd:.2f}, csd_deg: {csd * 180 / np.pi:.2f}',
                                                save_name=f'set_size_{set_sizes[i_ssize]}_seed_{seed}')

                w_diff_seeds.append(w)
                csd_diff_seeds.append(csd)
                aggregate_residuals.append(residuals)

            errors_all_ssizes_all_seeds.extend(errors_all_seeds)
            w_diff_ssizes.append(w_diff_seeds)
            csd_diff_ssizes.append(csd_diff_seeds)

            # for each set size, pool all seeds, fit mixture
            errors_all_seeds_centered = center_angle(copy.deepcopy(np.array(errors_all_seeds)) * np.pi / 180)
            w, kappa, csd = fit_von_mises_uniform_mixture(errors_all_seeds_centered)
            # plot histogram of errors, fit, and residuals
            density, mixture_fit, residuals = \
                get_density_fit_and_residual(errors_all_seeds_centered, x_axis, w, kappa)
            plots.plot_density_fit_residual(x_axis, density, mixture_fit, residuals,
                                            title=f'Set Size: {set_sizes[i_ssize]}, All seeds \n' +
                                                  f'w: {w:.2f}, kappa: {kappa:.2f} \n' +
                                                  f'csd: {csd:.2f}, csd_deg: {csd * 180 / np.pi:.2f}',
                                            save_name=f'set_size_{set_sizes[i_ssize]}_seed_all')

        # for all set sizes and all seeds, fit mixture
        errors_all_ssizes_all_seeds_centered = \
            center_angle(copy.deepcopy(np.array(errors_all_ssizes_all_seeds)) * np.pi / 180)
        w, kappa, csd = fit_von_mises_uniform_mixture(errors_all_ssizes_all_seeds_centered)
        # plot histogram of errors, fit, and residuals
        density, mixture_fit, residuals = \
            get_density_fit_and_residual(errors_all_ssizes_all_seeds_centered, x_axis, w, kappa)
        plots.plot_density_fit_residual(x_axis, density, mixture_fit, residuals,
                                        title=f'All set sizes, All seeds\n' +
                                              f'w: {w:.2f}, kappa: {kappa:.2f} \n' +
                                              f'csd: {csd:.2f}, csd_deg: {csd * 180 / np.pi:.2f}',
                                        save_name=f'set_size_all_seed_all')

        # aggregate residuals of each individual fit
        # across all set sizes and all seeds
        all_residuals = np.array(aggregate_residuals)
        all_residuals = all_residuals.transpose()
        all_residuals = all_residuals.reshape(-1, 6, 4).mean(axis=1)
        all_residuals = (all_residuals * (0.28 / np.pi * 180)).tolist() # change the scale so it is comparable to data (a 0.28 rad window)
        plots.error_plot(
            x_axis,
            [all_residuals],
            'Estimation error',
            'Density',
            fig_dir='continuous_report',
            fig_name=f'{model}_residuals_all_avg',
            mode='errorshade',
            legend=False
        )

        # plot how w changes with set size
        plots.error_plot(
            set_sizes,
            [w_diff_ssizes],
            'Set Size',
            'w',
            fig_dir='continuous_report',
            fig_name=f'{model}_error_dist' + '_w',
            mode='errorbar2',
            yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ylim=[0, 1.1],
            legend=False,
        )

        # plot how CSD changes with set size
        plots.error_plot(
            set_sizes,
            [csd_diff_ssizes],
            'Set Size',
            'CSD',
            fig_dir='continuous_report',
            fig_name=f'{model}_error_dist' + '_csd',
            mode='errorbar2',
            yticks=[0.2, 0.4, 0.6],
            ylim=[0, 0.7],
            legend=False,
        )    

def continuous_report_compare_norm_analysis():
    cfgs = experiments.continuous_report_compare_norm()

    set_sizes = [1, 2, 3, 4, 5, 6]
    norm_list = ['groupnorm', 'instancenorm', 'batchnorm', 'layernorm', ]
    performance = {}

    for k, norm in enumerate(norm_list):
        performance[norm] = get_continuous_report_performance_error(cfgs, k)[0]

    plots.continuous_report_plot(
        performance,
        set_sizes,
        norm_list,
        'recallSDvsSetSize_compare_normalizations',
        legend_fontsize=11,
        label_list=norm_list
    )

def continuous_report_input_noise_analysis():
    cfgs = experiments.continuous_report_input_noise()

    noise_list = [0, 0.02, 0.05, 0.1, 0.2, ]
    model_list = ['ResNet', 'pretResNet', ]
    set_sizes = cfgs.loc[0].config.num_patches

    for j, model in enumerate(model_list):
        performance = {}
        precision = {}

        for i, noise in enumerate(noise_list):
            performance[noise], errors, precision[noise] = get_continuous_report_performance_error(cfgs, j * len(noise_list) + i)

        plots.continuous_report_plot(
            performance, 
            set_sizes, 
            noise_list, 
            f'compare_input_noise_{model}',
            legend_title='Input noise std'
        )

def continuous_report_with_uncertainty_analysis():
    cfgs = experiments.continuous_report_with_uncertainty()
    model_list = ['pretResNet', ]

    for i, model in enumerate(model_list):
        recall_sd, errors, precision = get_continuous_report_performance_error(cfgs, i)
        errors = errors[0]

        # plot uncertainty distribution
        uncertainty = [[x[1] for x in y] for y in errors]
        plots.error_distribution_plot(
            (uncertainty, ), ['uncertainty', ],
            left=.05, right=.95, interval=.1, 
            plot_name=f'uncertainty_dist_{model}',
            plot_dir='continuous_report',
            x_label='$\sigma^2$',
            mode='errorbar1'
        )

        # plot error distributions with different confidence levels
        partition = [0.0, 0.08, 0.16, 0.24, 0.32, 0.4, 100]
        confidence_labels = ['5', '4', '3', '2', '1', '0']
        partitioned_error = []
        cmap = cm.get_cmap('cool')
        colors = [cmap(1-i) for i in np.linspace(0.1, 0.9, len(partition) - 1)]
        for j in range(len(partition) - 1):
            partitioned_error.append([])

            for single_model_error in errors:
                ret = []
                for error, sigma2 in single_model_error:
                    if sigma2 > partition[j] and sigma2 < partition[j + 1]:
                        ret.append(error)
                partitioned_error[-1].append(ret)

        confidence_labels = reversed(confidence_labels)
        partitioned_error = reversed(partitioned_error)

        plots.error_distribution_plot(
            partitioned_error,
            confidence_labels,
            left=-120, right=120, interval=20,
            plot_name=f'compare_confidence_dist_{model}',
            legend_title='Confidence',
            legend_loc='upper right',
            legend_fontsize=11,
            yticks=[0.0, 0.05, 0.1, 0.15, 0.2],
            colors=colors,
            coef=8
        )

        
def cued_continuous_report_analysis():
    train_cfgs = experiments.cued_continuous_report()
    test_cfgs = experiments.cued_continuous_report_test()

    att_list = ['feature', 'cbam', 'none', ]
    for j, att in enumerate(att_list):

        cued_performance = []
        uncued_performance = []
        cue_prob = [0.25, 0.5, 0.75, 1, ]

        for i, prob in enumerate(cue_prob):
            idx = i * len(att_list) + j

            std, errors, precision = get_continuous_report_performance_error(test_cfgs, idx * 2)
            uncued_performance.append(precision[0])

            std, errors, precision = get_continuous_report_performance_error(test_cfgs, idx * 2 + 1)
            cued_performance.append(precision[0])

        plots.error_plot(
            cue_prob,
            [cued_performance, uncued_performance],
            "Cue Validity",
            "Precision ($rad^{-1}$)",
            ['Cued', 'Uncued'],
            fig_dir='cued_continuous_report',
            fig_name=f'prob_precision_att{att}',
            mode='errorbar2',
            colors=['darkorchid', 'seagreen'],
        )


def cued_continuous_report_set_size_analysis():
    cfgs = experiments.cued_continuous_report_set_size()

    model_list = ['with attention', 'without attention', ]
    set_sizes = [1, 2, 3, 4, 5, 6]

    performance = {}
    precision = {}

    for i, model in enumerate(model_list):
        performance[model], errors, precision[model] = get_continuous_report_performance_error(cfgs, i)

    plots.continuous_report_plot(
        performance, 
        set_sizes, 
        model_list, 
        f'cued_compare_att',
        plot_dir='cued_continuous_report',
        legend_fontsize=11,
    )