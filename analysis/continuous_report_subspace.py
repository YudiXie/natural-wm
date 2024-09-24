import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.decomposition import PCA
from tasks.tasktools import angle2color
from configs.configs import ContinuousReportConfig
from configs.config_global import NP_SEED, TCH_SEED
from copy import deepcopy
from utils.config_utils import configs_df2config_dict
from collections import defaultdict
from tasks.tasktools import sample_color_angles
from tasks.continuousreport import ContinuousReport

import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles
import os
import torch

from analysis.plots import scatter_plot, plot_img
from analysis.activity import record_activity_ngym

def get_subspaces(X, Y, set_size, save_path=None, print_img=False, n_dots=60, subspace_dim=2):
    
    reg = Lasso(alpha=0.0001, max_iter=5000).fit(X, Y)
    print('Linear Regression Score = ', reg.score(X, Y))
    W: np.ndarray = reg.coef_.T # shape (set_size * n_dots, n_neurons)

    subspaces = []
    explained_variances = []
    pcas = []
    
    for rank in range(set_size):

        if print_img:
            pca = PCA(n_components=n_dots - 1)
            pca.fit(W[rank * n_dots: rank * n_dots + n_dots, ]) 

            cumulative_ratio = pca.explained_variance_ratio_
            for i in range(1, n_dots - 1):
                cumulative_ratio[i] += cumulative_ratio[i - 1]

            explained_variances.append(cumulative_ratio)

        pca = PCA(n_components=subspace_dim)
        W_new = pca.fit_transform(W[rank * n_dots: rank * n_dots + n_dots, ]) # n_dots * subspace_dim 

        print("variance ratio:", pca.explained_variance_ratio_)
        subspaces.append(pca.components_)
        pcas.append(pca)

        if print_img:

            lim = int(max(1, np.abs(W_new).max() * 1.3, np.abs(W_new).max() + 1))
            colors = [angle2color((i / n_dots * 2 - 1) * np.pi) for i in range(n_dots)]
            scatter_plot(
                W_new, 
                save_path, 
                f'set_size{set_size}_rank{rank + 1}', 
                xlim=lim, ylim=lim,
                xlabel='PC1', ylabel='PC2',
                colors=colors,
                output_to_fig_dir=False
            )

    return subspaces, explained_variances, pcas

def get_principal_angle(U: np.ndarray, V: np.ndarray):
    """
    get principle angle of two subspaces, U V should be of shape 2 * n
    """
    u, singular_values, vh = np.linalg.svd(np.matmul(U, V.T))
    # print(singular_values)
    return np.rad2deg(np.arccos(singular_values[0]))

def analyze(
    X, Y, 
    set_size, 
    save_path, 
    save_result=False, 
    concat_noise=0,
    noise_strength=0,
    eval=False,
    n_dots=60,
    return_subspaces=False
):

    print(X.shape, Y.shape) # (n_trials, set_size * ndots)

    mean = Y.mean(axis=0, keepdims=True)
    std = Y.std(axis=0, keepdims=True)
    Y = (Y - mean) / Y.std()

    Y += np.random.randn(*Y.shape) * noise_strength
    Y = np.concatenate((Y, np.random.randn(Y.shape[0], concat_noise)), axis=1)
    
    half = X.shape[0] // 2
    subspaces, ev, pcas = get_subspaces(X, Y, set_size, save_path, print_img=save_result, n_dots=n_dots)

    if save_result:
        torch.save((ev, pcas, (X, Y)), os.path.join(save_path, f'info_{set_size}.pth'))

    if return_subspaces:
        return subspaces, ev, pcas
    
    if set_size == 3:

        angle_between_ranks = (
            get_principal_angle(subspaces[0], subspaces[1]),
            get_principal_angle(subspaces[1], subspaces[2]),
            get_principal_angle(subspaces[0], subspaces[2])
        )

        if eval:
            return angle_between_ranks, ev, pcas

        first_half_subspaces = get_subspaces(X[: half], Y[: half], set_size, n_dots=n_dots)[0]
        second_half_subspaces = get_subspaces(X[half: ], Y[half: ], set_size, n_dots=n_dots)[0]

        angle_control = (
            get_principal_angle(first_half_subspaces[0], second_half_subspaces[0]),
            get_principal_angle(first_half_subspaces[1], second_half_subspaces[1]),
            get_principal_angle(first_half_subspaces[2], second_half_subspaces[2])
        )

        print(angle_between_ranks, angle_control)

        seqreproduction_angle_plot(
            angle_between_ranks,
            angle_control,
            save_path,
            f'set_size{set_size}_angle',
            plot_errorbar=False,
            append_fig_path=False
        )

        return angle_between_ranks, angle_control

def get_ev(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    X = (X - mean) / X.std()

    pca = PCA(n_components=X.shape[1] - 1)
    pca.fit(X) 

    cumulative_ratio = pca.explained_variance_ratio_
    dim = 1 / (cumulative_ratio ** 2).sum()
    for i in range(1, X.shape[1] - 1):
        cumulative_ratio[i] += cumulative_ratio[i - 1]

    return cumulative_ratio, dim 

# TODO: automatically determine the time step, instead of using hard-coded index
def store_activity(stored_activity: dict, recorded_activity: np.ndarray, layer_name: str, trial_info: dict):
    if layer_name == 'rnn':
        stored_activity[layer_name].append(recorded_activity[19].reshape(-1)) # Last step of delay
    elif layer_name == 'cnn':
        stored_activity[layer_name].append(recorded_activity[9].reshape(-1)) # Last step of sample
    else:
        stored_activity[layer_name].append(recorded_activity[0].reshape(-1))

def get_activities(config: ContinuousReportConfig, layer_list, record_batches, set_size=1, only_activity=False, **config_kwargs):

    config.num_workers = 1
    config.possible_colors = 30
    config.num_patches = (set_size, )

    for key, value in config_kwargs.items():
        config.__setattr__(key, value)

    activity, trial_info = \
        record_activity_ngym(
            config, layer_list, 
            record_batches, store_activity=store_activity
        )

    if only_activity:
        return activity

    features = []
    for trial in trial_info:
        feature = np.zeros((config.possible_colors * set_size, ))
        for rank, color_angle in enumerate(trial['color_angles']):
            color_id = round((color_angle + np.pi) / (2 * np.pi) * config.possible_colors)
            feature[rank * config.possible_colors + color_id] = 1
        features.append(feature)
    features = np.stack(features, axis=0)

    return features, activity
        
def continuous_report_encoding_space(cfgs, idx, layer_list = ['cnn', 'rnn', ], record_batches=10):

    cfgs = configs_df2config_dict(deepcopy(cfgs))
    num_seeds = len(cfgs)

    subspace_angles = {}
    control_angles = {}
    for layer in layer_list:
        subspace_angles[layer] = [[] for _ in range(3)]
        control_angles[layer] = [[] for _ in range(3)]

    dims = defaultdict(list)
    
    for seed in range(num_seeds):

        config: ContinuousReportConfig = cfgs[seed][idx]
        np.random.seed(NP_SEED + config.seed)
        torch.manual_seed(TCH_SEED + config.seed)

        set_size = 3
        features, activity = get_activities(config, layer_list, record_batches, set_size=set_size, fixed_positions=True)

        print('Activity recorded.', flush=True)

        for layer in layer_list:
            layer_activity = np.stack(activity[layer], axis=0)
            save_path = os.path.join(config.save_path, f'{layer}_subspace')
            
            angles, c_angles = analyze(
                features, layer_activity, 
                set_size, save_path, 
                save_result=True, 
                n_dots=config.possible_colors
            )
            for j, angle in enumerate(angles):
                subspace_angles[layer][j].append(angle)
                control_angles[layer][j].append(c_angles[j])

            ratio, dim = get_ev(layer_activity)
            dims[layer].append(dim)
            print(f'Dim {layer}: {dim}')

    return subspace_angles, control_angles, dims

"""
def continuous_report_visualize_interference(cfgs, idx, layer_list=['rnn', ], record_batches=10):

    cfgs = configs_df2config_dict(deepcopy(cfgs))
    num_seeds = len(cfgs)
    n_dots = 30

    for seed in range(num_seeds):

        config: ContinuousReportConfig = cfgs[seed][idx]
        np.random.seed(NP_SEED + config.seed)
        torch.manual_seed(TCH_SEED + config.seed)

        features, activity = get_activities(config, layer_list, record_batches, set_size=1, fixed_positions=True)

        print('Activity recorded.', flush=True)
        original_pcas = {}
        activity_means = {}
        activity_stds = {}

        for layer in layer_list:
            layer_activity = np.stack(activity[layer], axis=0)
            save_path = os.path.join(config.save_path, f'{layer}_subspace')
            
            pca = analyze(
                features, layer_activity, 
                1, save_path, 
                save_result=True, 
                n_dots=config.possible_colors,
                return_subspaces=True
            )[2][0]
            
            original_pcas[layer] = pca
            activity_means[layer] = layer_activity.mean(axis=0, keepdims=True)
            activity_stds[layer] = layer_activity.std()

        for angle_idx, angle2 in enumerate([0, -np.pi]):

            features, activity = \
                get_activities(
                    config, layer_list, record_batches, 
                    set_size=2, 
                    fixed_positions=True, 
                    fixed_color_angles=[None, angle2]
                )

            for layer in layer_list:
                Y = (activity[layer] - activity_means[layer]) / activity_stds[layer]
                X = features

                reg = Lasso(alpha=0.0001, max_iter=5000).fit(X, Y)
                print('Linear Regression Score = ', reg.score(X, Y))

                W: np.ndarray = reg.coef_.T
                W_new = original_pcas[layer].transform(W[: n_dots])

                lim = int(max(1, np.abs(W_new).max() * 1.3, np.abs(W_new).max() + 1))
                colors = [angle2color((i / n_dots * 2 - 1) * np.pi) for i in range(n_dots)]
                save_path = os.path.join(config.save_path, f'{layer}_subspace')

                scatter_plot(
                    W_new, 
                    save_path, 
                    f'interference_1_2_angle{angle_idx}', 
                    xlim=lim, ylim=lim,
                    xlabel='PC1', ylabel='PC2',
                    colors=colors,
                    output_to_fig_dir=False
                )
"""

def get_color_ring_subspace(
        config, 
        layer_list, 
        record_batches=1, 
        color_list=[None, ], 
        n_dots=30, 
        do_pca=False, 
        plot_idx=None,
        position_list=[0, 1, 2, 3, 4, 5, ]
    ):
    """
    Calculate the encoding subspace of a color ring
    :param layer_list: list of layers to analyze
    :param record_batches: number of batches to record for each color
    :param color_list: the color of each patch, one of the color angles should be None, 
        we will compute the encoding subspace for this patch
    :param plot_idx: if not None, manually enforce the index of the patch to compute the encoding subspace
    :param n_dots: number of points to sample on the color ring
    :param do_pca: whether to do pca analysis
    :param position_list: list of positions to sample
    returns a dict containing
    'activity': (n_dots, n_neurons), the mean population activity for each color
    'variance': (n_dots, n_neurons), the variance in population activity for each color when different input noise is added
    if do_pca is True, also contains
    'pca': the pca object from sklearn
    'dim': the dimension of the encoding subspace
    'ev': the explained variance ratio
    """

    positions = ContinuousReport.default_positions
    position_list = [positions[i] for i in position_list]

    set_size = len(color_list)
    plot_idx = None

    if plot_idx is None:
        for idx, color in enumerate(color_list):
            if color is None:
                plot_idx = idx
                break
        assert plot_idx is not None 

    activity_list = {}
    return_dict = {}
    variance_list = {}
    for layer in layer_list:
        activity_list[layer] = []
        variance_list[layer] = []
        return_dict[layer] = {}

    for i_color in range(n_dots):

        color = (i_color / n_dots * 2 - 1) * np.pi

        new_color_list = deepcopy(color_list)
        new_color_list[plot_idx] = color
        for idx, color in enumerate(new_color_list):
            if color is None:
                new_color_list[idx] = np.random.rand() * 2 * np.pi - np.pi

        activity = get_activities(
            config, 
            layer_list, 
            record_batches, 
            set_size=set_size, 
            fixed_positions=position_list, 
            fixed_color_angles=new_color_list,
            possible_colors=n_dots,
            only_activity=True
        )

        for layer in layer_list:
            raw_act = np.stack(activity[layer], axis=0)
            activity_list[layer].append(raw_act.mean(axis=0))
            variance_list[layer].append(raw_act.var(axis=0))

    for layer in layer_list:
        activity = np.stack(activity_list[layer], axis=0) # n_dots * n_neurons
        variance = np.stack(variance_list[layer], axis=0)

        return_dict[layer]['activity'] = activity
        return_dict[layer]['variance'] = variance

        if do_pca:
            n_neurons = activity.shape[1]
            n_components = min(n_neurons, n_dots)

            pca = PCA(n_components=n_components)
            pca.fit(activity)

            cumulative_ratio = pca.explained_variance_ratio_
            dim = 1 / (cumulative_ratio ** 2).sum()
            for i in range(1, n_components - 1):
                cumulative_ratio[i] += cumulative_ratio[i - 1]

            return_dict[layer]['dim'] = dim
            return_dict[layer]['ev'] = cumulative_ratio
            return_dict[layer]['pca'] = pca
        
    return return_dict

def new_list():
    return [[] for _ in range(6)]

def continuous_report_visualize_interference_set_size(config, layer_list=['rnn', ], num_iters=10, instances_per_dot=10):
    """
    Compute the strength of the signal and noise with varying set sizes
    Visualize the encoding subspace (2d PCA analysis)
    :param layer_list: list of layers to analyze
    :param record_batches: number of batches to record
    :param num_iters: number of iterations to run, colors are randomly sampled each time
    :return: a dictionary of various statistics: dict[str, dict[str, list]]
        {statistics-name: {layer_name: [0 ... set-sizes]}}
    """

    n_dots = 18
    return_dict_keys = ['signal', 'noise', 'snr', 'w_signal', 'w_noise', 'w_snr']
    return_dict = {}
    for key in return_dict_keys:
        return_dict[key] = {layer: [0] * 6 for layer in layer_list}

    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    config.batch_size = instances_per_dot

    for _iter in range(num_iters):

        color_list = sample_color_angles(5, possible_colors=n_dots)
        lims = {}
        subspaces = []

        for i in range(6):
            if _iter == 0:
                # Save the sample image
                image = ContinuousReport.get_image([None, ] + color_list[: i])
                image = torch.from_numpy(image)
                image_path = os.path.join(config.save_path, 'examples')
                plot_img(image, fig_path=image_path, fig_name=f'set_size_{i + 1}')

            subspace = get_color_ring_subspace(
                config, layer_list, 
                record_batches=1, 
                color_list=[None, ] + color_list[: i],
                n_dots=n_dots,
                do_pca=True
            )
            subspaces.append(subspace)
        pcas = {}

        for i in range(6):
            subspace = subspaces[i]

            for layer in layer_list:
                if i == 0:
                    pcas[layer] = subspace[layer]['pca']
                W = subspace[layer]['activity'] # (n_dots, n_neurons)
                
                signal = W.var(axis=0).sum()
                variance = subspace[layer]['variance'].mean(axis=0)
                noise = variance.sum()

                W_new = pcas[layer].transform(W) # (n_dots, min(n_dots, n_neurons))
                w_signal = (W_new.var(axis=0) * pcas[layer].explained_variance_ratio_).sum()

                _noise_var: np.ndarray = ((pcas[layer].components_ ** 2) * variance) # (min(n_dots, n_neurons), n_neurons)
                w_noise = (_noise_var.mean(axis=1) * pcas[layer].explained_variance_ratio_).sum()

                info_dict = {
                    'signal': signal,
                    'noise': noise,
                    'snr': signal / noise,
                    'w_signal': w_signal,
                    'w_noise': w_noise,
                    'w_snr': w_signal / w_noise
                }

                for key in return_dict_keys:
                    return_dict[key][layer][i] += info_dict[key] / num_iters

                print(f'iter {_iter}, {layer}, set_size {i + 1}: ', info_dict)

                if _iter == 0:
                    # Visualize the color encoding in the subspace computed when set_size = 1
                    W_new: np.ndarray = W_new[:, :2] # (n_dots, 2)
                    if i == 0:
                        lims[layer] = round(np.abs(W_new).max() * 13) / 10
                        lims[layer] = (-lims[layer], lims[layer])
                    
                    lim = lims[layer]
                    colors = [angle2color((c / n_dots * 2 - 1) * np.pi) for c in range(n_dots)]
                    save_path = os.path.join(config.save_path, f'{layer}_subspace')

                    def connect_dots(X, Y):
                        def _connect_dots(plt):
                            n = len(X)
                            for i in range(n):
                                plt.plot([X[i], X[(i + 1) % n]], [Y[i], Y[(i + 1) % n]], color='gray', linewidth=1)
                        return _connect_dots

                    scatter_plot(
                        W_new[:, 0], 
                        W_new[:, 1],
                        fig_dir=save_path, 
                        fig_name=f'subspace_set_size{i + 1}_global', 
                        xlim=lim, ylim=lim,
                        x_label='PC1', y_label='PC2',
                        colors=colors,
                        marker='o',
                        extra_lines=connect_dots(W_new[:, 0], W_new[:, 1]),
                        xticks=(lim[0], 0, lim[1]),
                        yticks=(lim[0], 0, lim[1])
                    )

                    # Visualize the color encoding in the subspace computed when set_size = i
                    W_new: np.ndarray = subspace[layer]['pca'].transform(W)[:, :2]
                    lim = lims[layer]

                    scatter_plot(
                        W_new[:, 0], 
                        W_new[:, 1],
                        fig_dir=save_path, 
                        fig_name=f'subspace_set_size{i + 1}_current', 
                        xlim=lim, ylim=lim,
                        x_label='PC1', y_label='PC2',
                        colors=colors,
                        marker='o',
                        extra_lines=connect_dots(W_new[:, 0], W_new[:, 1]),
                        xticks=(lim[0], 0, lim[1]),
                        yticks=(lim[0], 0, lim[1])
                    )

    torch.save(return_dict, os.path.join(config.save_path, 'encoding_subspace_snr.pth'))
    return return_dict

def continuous_report_visualize_interference(cfgs, idx, layer_list=['rnn', ], record_batches=1):
    """
    Visualize interference on color ring subspaces
    """

    cfgs = configs_df2config_dict(deepcopy(cfgs))
    num_seeds = len(cfgs)
    n_dots = 30

    for seed in range(num_seeds):

        config: ContinuousReportConfig = cfgs[seed][idx]
        np.random.seed(NP_SEED + config.seed)
        torch.manual_seed(TCH_SEED + config.seed)
        config.batch_size = 32

        pcas = {}
        lims = {}
        subspaces = defaultdict(list)

        for exp_idx in range(12):

            plot_idx = None
            color_list = []
            position_list = [0, 1, 2, 3, 4, 5,]

            if exp_idx < 2: 
                color_list = [None, ]
                position_list = [position_list[exp_idx], ]
            elif exp_idx < 4:
                color_list = [None, None, ]
                plot_idx = exp_idx - 2
            elif exp_idx < 8:
                color_list = [None, (exp_idx - 4) * 0.5 * np.pi - np.pi]
            elif exp_idx < 12:
                color_list = [(exp_idx - 8) * 0.5 * np.pi - np.pi, None]
            else:
                raise NotImplementedError
                
            positions = ContinuousReport.default_positions
            new_position_list = [positions[i] for i in position_list]
            image = ContinuousReport.get_image(color_list, positions=new_position_list)
            image = torch.from_numpy(image)
            image_path = os.path.join(config.save_path, 'examples')
            plot_img(image, fig_path=image_path, fig_name=f'exp_idx_{exp_idx}')

            return_dict = get_color_ring_subspace(
                config, layer_list, 
                record_batches=record_batches, 
                color_list=color_list,
                n_dots=n_dots,
                do_pca=True,
                position_list=position_list,
                plot_idx=plot_idx
            )

            for layer in layer_list:

                if exp_idx == 0:
                    pcas[layer] = return_dict[layer]['pca']
                    print('EV of first 2 PCs:', return_dict[layer]['ev'][1])

                W = return_dict[layer]['activity']
                W_new = pcas[layer].transform(W) # (n_dots, min(n_dots, n_neurons))
                W_new: np.ndarray = W_new[:, :2] # (n_dots, 2)
                subspaces[layer].append(return_dict[layer]['pca'].components_[:2])
                
                if exp_idx == 0:
                    lims[layer] = round(np.abs(W_new).max() * 130) / 100
                
                lim = lims[layer]
                colors = [angle2color((i / n_dots * 2 - 1) * np.pi) for i in range(n_dots)]
                save_path = os.path.join(config.save_path, f'{layer}_subspace')

                print(f"exp {exp_idx}, {layer}, dim = ", return_dict[layer]['dim'])

                scatter_plot(
                    W_new, 
                    save_path, 
                    f'subspace_exp_idx_{exp_idx}', 
                    xlim=lim, ylim=lim,
                    xlabel='PC1', ylabel='PC2',
                    colors=colors,
                    output_to_fig_dir=False
                )

        for layer in layer_list:

            pairs = [(0, 2), (1, 3), (0, 1), (2, 3)]
            pairs += [(i, i + 1) for i in range(4, 11)]
            pairs += [(i, i + 4) for i in range(4, 8)]

            for x, y in pairs:
                print(f'{layer} Angle ({x}, {y}): ', get_principal_angle(subspaces[layer][x], subspaces[layer][y]))

    return {}

def continuous_report_neural_dim(cfgs, idx, layer_list = ['cnn', 'rnn', ], record_batches=10, set_sizes=list(range(1, 7))):

    cfgs = configs_df2config_dict(deepcopy(cfgs))
    num_seeds = len(cfgs)
    dims = {}

    for layer in layer_list:
        dims[layer] = [[] for n in set_sizes]
       
    for i, set_size in enumerate(set_sizes):
        for seed in range(num_seeds):

            config: ContinuousReportConfig = deepcopy(cfgs[seed][idx])
            np.random.seed(NP_SEED + config.seed)
            torch.manual_seed(TCH_SEED + config.seed)

            features, activity = get_activities(config, layer_list, record_batches, set_size=set_size)
            print('Activity recorded.', flush=True)

            for layer in layer_list:
                layer_activity = np.stack(activity[layer], axis=0)
                ratio, dim = get_ev(layer_activity)
                dims[layer][i].append(dim)

                print(f'set_size: {set_size}, seed: {seed}, layer: {layer}, ratio:', ratio[: 10])

    return dims

def continuous_report_overall_activity(cfgs, idx, layer_list = ['cnn', 'rnn', ], record_batches=10, set_sizes=list(range(1, 7))):

    cfgs = configs_df2config_dict(deepcopy(cfgs))
    num_seeds = len(cfgs)
    mean_activity_list = {}

    for layer in layer_list:
        mean_activity_list[layer] = [[] for n in set_sizes]   
    
    for seed in range(num_seeds):
        layer_activities = {}
        for layer in layer_list:
            layer_activities[layer] = []

        for i, set_size in enumerate(set_sizes):
            config: ContinuousReportConfig = deepcopy(cfgs[seed][idx])
            np.random.seed(NP_SEED + config.seed)
            torch.manual_seed(TCH_SEED + config.seed)

            features, activity = get_activities(config, layer_list, record_batches, set_size=set_size)
            print('Activity recorded.', flush=True)

            for layer in layer_list:
                layer_activity = np.stack(activity[layer], axis=0)
                layer_activities[layer].append(layer_activity)

        mean_activity = {}
        std_activity = {}
        for layer in layer_list:
            activity = np.concatenate(layer_activities[layer], axis=0)
            mean_activity[layer] = activity.mean(axis=0)
            std_activity[layer] = activity.std(axis=0) + 1e-6

        for i, set_size in enumerate(set_sizes):
            for layer in layer_list:
                layer_activity = (layer_activities[layer][i] - mean_activity[layer]) / std_activity[layer]
                mean = layer_activity.mean()
                std = layer_activity.std()
                mean_activity_list[layer][i].append(mean)

                print(f'set_size: {set_size}, seed: {seed}, layer: {layer}, mean: {mean}, std: {std}')

    return mean_activity_list