import configs.experiments as experiments
import analysis.plots as plots
import matplotlib.colors as mcolors
from analysis.plots import make_rgb_with_alpha
from configs.exp_analysis import get_luck_vogel_performance, get_continuous_report_performance_error

__all__ = [
    'continuous_report_noise_analysis',
    'luckvogel_noise_analysis',
    'luckvogel_input_and_additive_rnn_noise_analysis',
    'luckvogel_input_noise_train_analysis',
    'continuous_report_input_noise_train_analysis',
]

def noise_plot(noise_list, noise_name, main_color, performance, model_name, task_name='luckvogel', legend=True):

    n = len(noise_list)
    color_alpha = [1 - (n - i - 1) * 0.2 for i in range(n)]
    colors = [make_rgb_with_alpha(main_color, a) for a in color_alpha]
    alphas = [0.9] * n

    if task_name == 'luckvogel':
        plots.luckvogel_errorbar_plot(
            performance,
            [1, 2, 3, 4, 8, 12],
            noise_list,
            f'{task_name}_{noise_name}_{model_name}',
            legend_title='noise std',
            chance_level_label=None,
            errormode='sem',
            mode='errorbar3',
            colors=colors,
            alphas=alphas,
            plot_dir='noise',
            xticks=(0, 4, 8, 12),
            legend=legend,
            legend_bbox_to_anchor=(1.05, 1),
            legend_loc='upper left',
            legend_fontsize=12,
            figsize=(4.2, 3)
        )

    elif task_name == 'continuous_report':
        plots.continuous_report_plot(
            performance,
            [1, 2, 3, 4, 5, 6],
            noise_list,
            f'{task_name}_{noise_name}_{model_name}',
            legend_title='noise std',
            errormode='sem',
            mode='errorbar3',
            colors=colors,
            alphas=alphas,
            plot_dir='noise',
            legend=legend,
            legend_bbox_to_anchor=(1.05, 1),
            legend_loc='upper left',
            legend_fontsize=12,
            figsize=(4.2, 3)
        )

def luckvogel_noise_analysis():
    cfgs = experiments.luckvogel_additive_rnn_noise_test()
    noise_list = [0.1, 0.3, 0.5, 0.7, ]
    model_list = ['ResNet', 'pretResNet', ]

    for j, model in enumerate(model_list):
        performance = {}
        for i, noise in enumerate(noise_list):
            performance[noise] = get_luck_vogel_performance(cfgs, j * len(noise_list) + i)

        noise_plot(
            noise_list,
            'additive_rnn_noise',
            mcolors.to_rgb('C0'),
            performance,
            model,
            task_name='luckvogel'
        )

    cfgs = experiments.luckvogel_multiplicative_rnn_noise_test()
    noise_list = [0, 0.1, 0.2, 0.3, ]
    for j, model in enumerate(model_list):
        performance = {}
        for i, noise in enumerate(noise_list):
            performance[noise] = get_luck_vogel_performance(cfgs, j * len(noise_list) + i)

        noise_plot(
            noise_list,
            'multiplicative_rnn_noise',
            mcolors.to_rgb('C1'),
            performance,
            model,
            task_name='luckvogel'
        )

    cfgs = experiments.luckvogel_input_noise_test()
    noise_list = [0.1, 0.2, 0.3, 0.4, ]
    for j, model in enumerate(model_list):
        performance = {}
        for i, noise in enumerate(noise_list):
            performance[noise] = get_luck_vogel_performance(cfgs, j * len(noise_list) + i)

        noise_plot(
            noise_list,
            'input_noise',
            mcolors.to_rgb('C2'),
            performance,
            model,
            task_name='luckvogel'
        )

def continuous_report_noise_analysis():
    cfgs = experiments.continuous_report_additive_rnn_noise_test()
    noise_list = [0.1, 0.3, 0.5, 0.7]
    model_list = ['ResNet', 'pretResNet', ]

    for j, model in enumerate(model_list):
        performance = {}
        precision = {}
        for i, noise in enumerate(noise_list):
            performance[noise], errors, precision[noise] = get_continuous_report_performance_error(cfgs, j * len(noise_list) + i)

        noise_plot(
            noise_list,
            'additive_rnn_noise',
            mcolors.to_rgb('C0'),
            performance,
            model,
            task_name='continuous_report'
        )

    cfgs = experiments.continuous_report_multiplicative_rnn_noise_test()
    noise_list = [0, 0.1, 0.2, 0.3, ]
    for j, model in enumerate(model_list):
        performance = {}
        precision = {}
        for i, noise in enumerate(noise_list):
            performance[noise], errors, precision[noise] = get_continuous_report_performance_error(cfgs, j * len(noise_list) + i)

        noise_plot(
            noise_list,
            'multiplicative_rnn_noise',
            mcolors.to_rgb('C1'),
            performance,
            model,
            task_name='continuous_report'
        )

    cfgs = experiments.continuous_report_input_noise_test()
    noise_list = [0.1, 0.2, 0.3, 0.4, ]
    for j, model in enumerate(model_list):
        performance = {}
        precision = {}
        for i, noise in enumerate(noise_list):
            performance[noise], errors, precision[noise] = get_continuous_report_performance_error(cfgs, j * len(noise_list) + i)

        noise_plot(
            noise_list,
            'input_noise',
            mcolors.to_rgb('C2'),
            performance,
            model,
            task_name='continuous_report'
        )

def luckvogel_input_and_additive_rnn_noise_analysis():
    inp_noise_cfgs = experiments.luckvogel_input_noise_test()
    inp_noise_list = [0.1, 0.3, 0.5]
    rnn_noise_cfgs = experiments.luckvogel_additive_rnn_noise_test()
    rnn_noise_noise_list = [0.03, 0.06, 0.1]

    color1 = mcolors.to_rgb('darkorchid')
    color2 = mcolors.to_rgb('seagreen')
    color_alpha = [0.6, 0.8, 1.0]
    colors = [make_rgb_with_alpha(color1, a) for a in color_alpha] + \
             [make_rgb_with_alpha(color2, a) for a in color_alpha]
    alphas = [0.9] * 6

    for i, model in enumerate(['ResNet', 'pretResNet', ]):
        noise_performance = {}
        all_noise_labels = []
        for j, noise in enumerate(inp_noise_list):
            noise_performance[f'input {noise}'] = get_luck_vogel_performance(inp_noise_cfgs, i * len(inp_noise_list) + j)
            all_noise_labels.append(f'input {noise}')
        for j, noise in enumerate(rnn_noise_noise_list):
            noise_performance[f'rnn {noise}'] = get_luck_vogel_performance(rnn_noise_cfgs, i * len(rnn_noise_noise_list) + j)
            all_noise_labels.append(f'rnn {noise}')

        plots.luckvogel_errorbar_plot(
            noise_performance,
            [1, 2, 3, 4, 8, 12],
            all_noise_labels,
            f'input_and_additive_rnn_noise_{model}',
            legend_title='Noise std',
            chance_level_label=None,
            errormode='sem',
            mode='errorbar3',
            colors=colors,
            alphas=alphas,
            xticks=(0, 4, 8, 12),
            legend=False,
        )

def luckvogel_input_noise_train_analysis():
    cfgs = experiments.luckvogel_input_noise_train()
    model_list = ['Class_CIFAR10', ]
    noise_list = [0, 0.05, 0.1, 0.2]

    for j, model in enumerate(model_list):
        performance = {}
        for i, noise in enumerate(noise_list):
            performance[noise] = get_luck_vogel_performance(cfgs, j * len(noise_list) + i)

        noise_plot(
            noise_list,
            f'input_noise_train_{model}',
            mcolors.to_rgb('darkorchid'),
            performance,
            model,
            task_name='luckvogel',
            legend=True
        )

def continuous_report_input_noise_train_analysis():
    inp_noise_cfgs = experiments.continuous_report_input_noise_train()
    model_list = ['Class_CIFAR10', ]
    noise_list = [0, 0.05, 0.1, 0.2]

    for j, model in enumerate(model_list):
        performance = {}
        precision = {}
        for i, noise in enumerate(noise_list):
            performance[noise], errors, precision[noise] = get_continuous_report_performance_error(inp_noise_cfgs, j * len(noise_list) + i)

        noise_plot(
            noise_list,
            f'input_noise_train_{model}',
            mcolors.to_rgb('darkorchid'),
            performance,
            model,
            task_name='continuous_report',
            legend=True
        )