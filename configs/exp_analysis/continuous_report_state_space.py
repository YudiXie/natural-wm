import torch
import numpy as np
import analysis.plots as plots
from collections import defaultdict
import configs.experiments as experiments

__all__ = [
    'continuous_report_subspace_analysis',
    'continuous_report_encoding_subspace_vs_setsize_compare_norm_analysis',
    'continuous_report_encoding_subspace_vs_setsize_analysis',
    'continuous_report_visualize_interference_analysis',
    'continuous_report_neural_dim_analysis',
    'continuous_report_overall_activity_analysis',
    'continuous_report_overall_activity_compare_norm_analysis',
]

def continuous_report_subspace_analysis():
    from analysis.continuous_report_subspace import continuous_report_encoding_space

    layer_list = ['cnn', 'rnn', ]
    model_list = ['ResNet', 'pretResNet']
    cfgs = experiments.continuous_report_ngym()
    dims = [[] for model in model_list]

    for i, model in enumerate(model_list):
        subspace_angles, control_angles, neural_dims = continuous_report_encoding_space(cfgs, i, layer_list=layer_list)
        
        for layer in layer_list:
            dims[i].append(neural_dims[layer])

        for layer in layer_list:
            plots.seqreproduction_angle_plot(
                subspace_angles[layer],
                control_angles[layer],
                'continuous_report_state_space',
                f'subspace_angles_set_size3_{model}_{layer}'
            )

    plots.bar_plot(
        dims, layer_list, 'Dim', model_list, 
        fig_dir='continuous_report_state_space',
        fig_name='compare_pret_nonpret_dim_set_size_3'
    )

def continuous_report_encoding_subspace_vs_setsize_compare_norm_analysis():

    layer_list = ['cnn', 'rnn', ]
    model_list = ['groupnorm', 'instancenorm', 'batchnorm', 'layernorm', ]
    set_size = list(range(1, 7))
    collect_data = True

    if collect_data:
        from analysis.continuous_report_subspace import \
                continuous_report_visualize_interference_set_size

        cfgs = experiments.continuous_report_compare_norm()
        return_dicts = defaultdict(list)

        for i, model in enumerate(model_list):
            return_dict = continuous_report_visualize_interference_set_size(cfgs, i, layer_list=layer_list, num_iters=10)

            for key, value in return_dict.items():
                return_dicts[key].append(value)

        torch.save(return_dicts, 'experiments/continuous_report_compare_norm/encoding_space_stat_cache.pth')
    else:
        return_dicts = torch.load('experiments/continuous_report_compare_norm/encoding_space_stat_cache.pth')

    for layer in layer_list:
        att_list = ['signal_var', 'log_snr', ]

        for att in att_list:

            if att[-3:] != 'snr':
                plots.error_plot(
                    set_size,
                    [return_dicts[att][i][layer] for i in range(len(model_list))],
                    'Set Size', 'Variance', 
                    model_list,
                    fig_dir='continuous_report_state_space',
                    fig_name=f'{layer}_{att}_compare_norm',
                    errormode='sem'
                )
            
            else:
                plots.error_plot(
                    set_size,
                    [return_dicts[att][i][layer] for i in range(len(model_list))],
                    'Set Size', 'log10(SNR)', 
                    model_list,
                    fig_dir='continuous_report_state_space',
                    fig_name=f'{layer}_{att}_compare_norm',
                    errormode='sem'
                )

    return return_dicts

def continuous_report_encoding_subspace_vs_setsize_analysis():

    layer_list = ['cnn', 'rnn', ]
    model_list = ['ResNet', 'pretResNet']
    set_size = list(range(1, 7))
    collect_data = True

    cfgs = experiments.continuous_report_ngym()

    if collect_data:
        from analysis.continuous_report_subspace import \
                continuous_report_visualize_interference_set_size

        cfgs = experiments.continuous_report_ngym()
        return_dicts = []

        for idx in range(len(model_list)):
            return_dict = {}
            return_dict_keys = ['signal', 'noise', 'snr', 'w_signal', 'w_noise', 'w_snr', 'log_snr', 'log_w_snr']
            for key in return_dict_keys:
                return_dict[key] = {layer: [[] for _ in range(6)] for layer in layer_list}
            
            for seed in range(4):
                cfg = cfgs.loc[seed * 2 + idx].config
                assert cfg.seed == seed
                info = continuous_report_visualize_interference_set_size(cfg, layer_list=layer_list, num_iters=10)

                for key in info.keys():
                    for layer in layer_list:
                        for n in set_size:
                            return_dict[key][layer][n].append(info[key][layer][n])
                            if key[-3: ] == 'snr':
                                return_dict['log_' + key][layer][n].append(np.log(info[key][layer][n]))
            
            return_dicts.append(return_dict)

        torch.save(return_dicts, 'experiments/continuous_report_ngym/encoding_space_stat_cache.pth')
    else:
        return_dicts = torch.load('experiments/continuous_report_ngym/encoding_space_stat_cache.pth')

    for layer in layer_list:
        att_list = ['signal', 'noise', 'log_snr', 'log_w_snr']

        for att in att_list:
            if att[-3:] != 'snr':
                plots.error_plot(
                    set_size,
                    [return_dicts[1][att][layer], ],
                    'Set Size', 'Variance', 
                    ['Pre-trained', ],
                    colors=['C0', ],
                    fig_dir='continuous_report_state_space',
                    fig_name=f'{layer}_{att}_pretrained',
                    errormode='sem',
                    ylim=[0, None]
                )

                plots.error_plot(
                    set_size,
                    [return_dicts[0][att][layer], ],
                    'Set Size', 'Variance', 
                    ['End-to-end', ],
                    colors=['C1', ],
                    fig_dir='continuous_report_state_space',
                    fig_name=f'{layer}_{att}_endtoend',
                    errormode='sem',
                    ylim=[0, None]
                )

            else:
                plots.error_plot(
                    set_size,
                    [return_dicts[1][att][layer], return_dicts[0][att][layer], ],
                    'Set Size', 'log(SNR)', 
                    ['Pre-trained', 'End-to-end', ],
                    colors=['C0', 'C1', ],
                    fig_dir='continuous_report_state_space',
                    fig_name=f'{layer}_{att}',
                    errormode='sem'
                )

    return return_dicts

def continuous_report_visualize_interference_analysis():

    layer_list = ['cnn', 'rnn', ]
    model_list = ['ResNet', 'pretResNet']
    set_size = list(range(1, 7))

    from analysis.continuous_report_subspace import continuous_report_visualize_interference

    cfgs = experiments.continuous_report_ngym()
    return_dicts = defaultdict(list)

    for i, model in enumerate(model_list):
        continuous_report_visualize_interference(cfgs, i, layer_list=layer_list)

def continuous_report_neural_dim_analysis():
    from analysis.continuous_report_subspace import continuous_report_neural_dim

    layer_list = ['cnn', 'rnn', ]
    model_list = ['End-to-end', 'Pre-trained']
    set_sizes = list(range(1, 7))
    cfgs = experiments.continuous_report_ngym()

    neural_dims = {}
    for layer in layer_list:
        neural_dims[layer] = []
    
    for i, model in enumerate(model_list):
        dims = continuous_report_neural_dim(cfgs, i, layer_list=layer_list, set_sizes=set_sizes, record_batches=10)

        for layer in layer_list:
            neural_dims[layer].append(dims[layer])

    for layer in layer_list:
        colors = ['#5AA9E6', '#FF6392']
        for i, (color, model) in enumerate(zip(colors, model_list)):
            plots.error_plot(
                set_sizes,
                [neural_dims[layer][i], ],
                'Set Size',
                'Dim',
                label_list=[model, ],
                fig_dir='continuous_report_state_space',
                fig_name=f'neural_dim_{layer}_{model}',
                colors=[color, ],
                ylim=(0, 18),
            )

def continuous_report_overall_activity_analysis():
    from analysis.continuous_report_subspace import continuous_report_overall_activity

    layer_list = ['resblock1', 'resblock2', 'resblock3', 'cnn', 'rnn', ]
    model_list = ['End-to-end', 'Pre-trained']
    set_sizes = list(range(1, 7))
    cfgs = experiments.continuous_report_ngym()

    mean_activations = {}
    for layer in layer_list:
        mean_activations[layer] = []
    
    for i, model in enumerate(model_list):
        act = continuous_report_overall_activity(cfgs, i, layer_list=layer_list, set_sizes=set_sizes, record_batches=10)
        for layer in layer_list:
            mean_activations[layer].append(act[layer])

    for layer in layer_list:
        colors = ['#5AA9E6', '#FF6392']
        
        plots.error_plot(
            set_sizes,
            mean_activations[layer],
            'Set Size',
            'Mean Activity',
            label_list=model_list,
            fig_dir='continuous_report_state_space',
            fig_name=f'overall_activity_{layer}',
            colors=colors,
        )

def continuous_report_overall_activity_compare_norm_analysis():
    from analysis.continuous_report_subspace import continuous_report_overall_activity

    layer_list = ['resblock1', 'resblock2', 'resblock3', 'rnn', ]
    model_list = ['groupnorm', 'instancenorm', 'batchnorm', 'layernorm', ]
    set_sizes = list(range(1, 7))
    cfgs = experiments.continuous_report_compare_norm()

    mean_activations = {}
    for layer in layer_list:
        mean_activations[layer] = []
    
    for i, model in enumerate(model_list):
        dims = continuous_report_overall_activity(cfgs, i, layer_list=layer_list, set_sizes=set_sizes, record_batches=10)
        for layer in layer_list:
            mean_activations[layer].append(dims[layer])

    for layer in layer_list:
        plots.error_plot(
            set_sizes,
            mean_activations[layer],
            'Set Size',
            'Mean Activity',
            label_list=model_list,
            fig_dir='continuous_report_state_space',
            fig_name=f'overall_activity_compare_norm_{layer}'
        )