import numpy as np
import torch
import analysis.plots as plots
import os.path as osp
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import analysis.plots as plots

from collections import defaultdict
from analysis.activity import record_activity_ngym
from analysis.activity_decoding import train_decoder
from configs.configs import LuckVogelConfig
from utils.config_utils import configs_df2config_dict
from copy import deepcopy

def store_activity(stored_activity, activity: np.ndarray, layer, trial_info):
    if layer in ['cnn', 'resblock1', 'resblock2', 'resblock3']:
        activity = (activity[: 10] - activity[-10: ]).mean(axis=0)
        if layer != 'cnn':
            activity = F.adaptive_avg_pool2d(torch.from_numpy(activity), (8, 8)).numpy()
        activity = activity.reshape(-1)
    elif layer in ['rnn', 'out_layer']:
        activity = activity[-1].reshape(-1)
    else:
        raise NotImplementedError(layer)
    stored_activity[layer].append(activity)
    return activity

def store_distance(stored_activity, activity: np.ndarray, layer, trial_info):
    if layer in ['cnn', 'resblock1', 'resblock2', 'resblock3']:
        activity = (activity[: 10] - activity[-10: ]).mean(axis=0)
        activity = (activity ** 2).sum() ** 0.5
    else:
        raise NotImplementedError(layer)
    stored_activity[layer].append(activity)
    return activity

def luck_vogel_decoding(
    config: LuckVogelConfig, 
    n_iter=1,
    test_batch=50,
    decoder_iter_num=200,
    decoder_log_every=50,
    record_layers=['cnn', 'rnn', ]
):
    """
    For each set size, first collect neural data from different layers, then train a MLP decoder for each layer
    If n_iter > 1, then we collect neural data for n_iter times train the decoder for decoder_iter_num steps for each iteration
    Results are saved in config.save_path/decoder_acc.pth, can be retrieved by get_luck_vogel_decoder_performance
    :param test_batch: number of batches to collect neural data
    :param decoder_iter_num: number of steps for decoder training
    :param decoder_log_every: log decoder performance every n steps
    :param record_layers: list of layers to record neural data from
    """

    set_sizes = config.num_patches
    best_accs = {}
    record_accs = {}

    for layer in record_layers:
        best_accs[layer] = [[] for n in set_sizes]
        record_accs[layer] = [[] for n in set_sizes]

    decoders = []
    for idx, n in enumerate(set_sizes):

        decoder = {}
        logger = {}

        for layer in record_layers:
            decoder[layer] = None
            logger[layer] = None

        for _iter in range(n_iter):
            config.num_patches = (n, )
            print("num_patches", config.num_patches, "iter ", _iter, flush=True)

            all_activity, trial_info = record_activity_ngym(
                config, 
                layers=record_layers, 
                test_batch=test_batch,
                store_activity=store_activity
            )

            label = [trial['ground_truth'] - 1 for trial in trial_info]
            label = np.array(label).astype(np.int32)

            for layer in record_layers:
                activity = np.stack(all_activity[layer], axis=0).astype(np.float32)
                print('neural activity and label shape: ', activity.shape, label.shape, flush=True)
                
                decoder[layer], logger[layer], best_acc = train_decoder(
                    config, activity, label, 
                    dict(n_hidden_units=512, n_hidden_layers=1),
                    logger_name=f'decoder_{n}_{layer}.txt',
                    iter_num=decoder_iter_num,
                    log_every=decoder_log_every,
                    decoder=decoder[layer],
                    logger=logger[layer]
                )

                best_accs[layer][idx].append(best_acc / 100)
        for layer in record_layers:
            best_accs[layer][idx] = [max(best_accs[layer][idx]), ]
        decoders.append(decoder)

    config.num_patches = set_sizes
    print("Decoding Result:", best_accs)
    torch.save((best_accs, record_accs), osp.join(config.save_path, f"decoder_acc.pth"))
    torch.save(decoders, osp.join(config.save_path, f"decoders.pth"))

def get_luck_vogel_decoder_performance(cfgs, idx, layer='cnn'):
    """
    Get the performance of the decoder trained on the neural data of the given layer
    """
    cfgs = configs_df2config_dict(cfgs)
    num_seeds = len(cfgs)
    set_sizes = cfgs[0][idx].num_patches
    performance = [[] for _ in set_sizes]

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        try:
            file_path = osp.join(cfg.save_path, f'decoder_acc.pth')
            decoder_acc, record_acc = torch.load(file_path)
            for accs, acc in zip(performance, decoder_acc[layer]):
                accs.append(np.mean(acc))
        except:
            print('Error occur when reading data at', cfg.save_path)

    return performance

def get_luck_vogel_similarity_scores(cfgs, idx, layer='cnn'):
    cfgs = configs_df2config_dict(cfgs)
    num_seeds = len(cfgs)
    set_sizes = cfgs[0][idx].num_patches
    similarity_list = [[] for _ in set_sizes]
    baseline_list = [[] for _ in set_sizes]

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]
        try:
            file_path = osp.join(cfg.save_path, f'decoder_sim.pth')
            similarity, baseline = torch.load(file_path)
            for sim_list, base_list, sim, base in zip(similarity_list, baseline_list, similarity[layer], baseline[layer]):
                sim_list.append(sim)
                base_list.append(base)
        except:
            print('Error occur when reading data at', cfg.save_path)

    return similarity_list, baseline_list

def get_luck_vogel_embedding_distances(configs, idx, layer='cnn'):
    """
    Get the distances between the embedding of the first and the last patch
    """
    cfgs = configs_df2config_dict(configs)
    num_seeds = len(cfgs)
    set_sizes = cfgs[0][idx].num_patches
    return_dict = {
        'signal': [[] for _ in set_sizes],
        'noise': [[] for _ in set_sizes],
        'snr': [[] for _ in set_sizes],
        'log_snr': [[] for _ in set_sizes],
    }

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]
        try:
            file_path = osp.join(cfg.save_path, f'embed_dist.pth')
            signal, noise, snr = torch.load(file_path)
            for i, n in enumerate(set_sizes):
                return_dict['signal'][i].append(signal[layer][i])
                return_dict['noise'][i].append(noise[layer][i])
                return_dict['snr'][i].append(snr[layer][i])
                return_dict['log_snr'][i].append(np.log(snr[layer][i]))
        except:
            print('Error occur when reading data at', cfg.save_path)

    return return_dict

def get_prob(x: np.ndarray):
    """
    Get probability from logits for an 2d numpy array
    """
    x = np.exp(x)
    return (x / np.sum(x, axis=-1, keepdims=True))[:, 0]

def get_score(decoder: torch.nn.Module, activity: np.ndarray):
    """
    Get decoding scores for each trial
    """
    n = activity.shape[0]
    bsz = 64
    scores = []
    for batch in range((n - 1) // bsz + 1):
        with torch.no_grad():
            logits = decoder(torch.from_numpy(activity[batch * bsz: (batch + 1) * bsz]).cuda())
            score = torch.softmax(logits, dim=-1)[:, 0]
            scores.append(score.cpu().numpy())
    return np.concatenate(scores, axis=0)

def scatter_plot(save_dir, layers, layer_names, scores_list, model_score_list, set_sizes, n_dots=50):
    """
    Scatter plot for decoding score vs. model score
    """
    n = len(set_sizes)
    # generate n blue colors from light to dark
    cmap = mpl.cm.get_cmap('Blues')
    colors = [cmap(i) for i in np.linspace(0.2, 0.8, n)]

    scores_list = {layer: np.concatenate([x[: n_dots] for x in scores_list[layer]], axis=0) for layer in layers}
    model_score_list = np.concatenate([x[: n_dots] for x in model_score_list], axis=0)

    for layer, layer_name in zip(layers, layer_names):
        X = scores_list[layer]
        Y = model_score_list
        C = [colors[i // (len(X) // n)] for i in range(len(X))]

        def reference_line(plt):
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)

        plots.scatter_plot(
            X, Y,
            x_label=f'Decoding from {layer_name}',
            y_label='Model Prediction',
            fig_dir=save_dir,
            fig_name=f'{layer}_decoder_vs_model',
            colors=C,
            dot_sizes=3,
            xlim=(-0.1, 1.1),
            ylim=(-0.1, 1.1),
            xticks=[0, 0.5, 1],
            yticks=[0, 0.5, 1],
            extra_lines=reference_line
        )

def get_similarities(model_scores, layer_scores, labels, layers):
    """
    For each layer and each set size, calculate the similarity between model scores and decoding scores
    In additon, return the baseline similarity between model scores and decoding scores if the decoding scores are shuffled
    """
    baseline = {}
    similarities = {}
    n = len(model_scores)
    for layer in layers:
        similarities[layer] = []
        baseline[layer] = []

        for i in range(n):
            X = (model_scores[i] > 0.5) == labels[i]
            Y = (layer_scores[layer][i] > 0.5) == labels[i]

            similarities[layer].append(np.mean(X == Y))

            T_indices = labels[i] == 1
            F_indices = labels[i] == 0
            # shuffle the decoding scores for T trials and F trials separately
            X = np.concatenate([X[T_indices], X[F_indices]])
            Y = np.concatenate([Y[np.random.permutation(T_indices)], Y[np.random.permutation(F_indices)]])
            baseline[layer].append(np.mean(X == Y))
    return similarities, baseline

def compare_decoder_model_behavior(
    config,
    test_batch=10,
    record_layers=['cnn', 'rnn', ],
    layer_names=['CNN', 'RNN', ],
):
    set_sizes = config.num_patches
    decoders = torch.load(osp.join(config.save_path, f"decoders.pth"))
    corr_matrices = []

    scores_list = defaultdict(list)
    model_score_list = []
    label_list = []

    for idx, n in enumerate(set_sizes):
        eval_config = deepcopy(config)
        eval_config.num_patches = (n, )
        eval_config.num_workers = 0
        all_activity, trial_info = record_activity_ngym(
            eval_config, 
            layers=record_layers + ['out_layer'], 
            test_batch=test_batch,
            store_activity=store_activity,
        )
        print("num_patches", config.num_patches, flush=True)

        label = [trial['ground_truth'] - 1 for trial in trial_info]
        label = np.array(label).astype(np.int32)
        label_list.append(label)

        scores = {}
        for layer in record_layers:
            activity = np.stack(all_activity[layer], axis=0).astype(np.float32)
            scores[layer] = get_score(decoders[idx][layer], activity)
            scores_list[layer].append(scores[layer])

        out_logits = np.stack(all_activity['out_layer'], axis=0).astype(np.float32)
        model_score = get_prob(out_logits[:, 1:])
        model_score_list.append(model_score)

        data_mat = np.stack([scores[layer] for layer in record_layers] + [model_score], axis=0)
        corr_matrices.append(np.corrcoef(data_mat))
        print("Correlation Matrix: ", corr_matrices, flush=True)

    scatter_plot(config.save_path, record_layers, layer_names, scores_list, model_score_list, set_sizes)
    similarity_score, baseline = get_similarities(model_score_list, scores_list, label_list, record_layers)
    torch.save((similarity_score, baseline), osp.join(config.save_path, f'decoder_sim.pth'))


def compute_embedding_distances(
    config,
    test_batch=10,
    record_layers=['cnn', ],
):
    
    set_sizes = config.num_patches
    signals = defaultdict(list)
    noises = defaultdict(list)
    snrs = defaultdict(list)

    for idx, n in enumerate(set_sizes):
        eval_config: LuckVogelConfig = deepcopy(config)
        eval_config.num_patches = (n, )
        eval_config.att = 'none'
        eval_config.input_noise = 0.1
        eval_config.strict_loading = False
        eval_config.num_workers = 1
        all_activity, trial_info = record_activity_ngym(
            eval_config, 
            layers=record_layers, 
            test_batch=test_batch,
            store_activity=store_distance,
        )

        label = [trial['ground_truth'] - 1 for trial in trial_info]
        label = np.array(label).astype(np.bool_)
        # only consider change trials (label = 1)
        
        for layer in record_layers:
            dist = np.array(all_activity[layer]).astype(np.float32)

            signal = np.mean(dist[label])
            noise = np.mean(dist[~label])
            snr = signal / noise

            # print(layer, n, np.mean(dist[label]), np.median(dist[label]), np.mean(dist[~label]), np.median(dist[~label]))
            signals[layer].append(signal)
            noises[layer].append(noise)
            snrs[layer].append(snr)

            a, b = np.max(dist[label]), np.min(dist[~label])

            plots.error_distribution_plot(
                [[np.log(dist[label])], [np.log(dist[~label])]],
                ['Different stimuli', 'Same stimuli'],
                left=0.025 * a, right=0.975 * a, interval=a / 20,
                x_label='Embedding Distance',
                y_label='Density',
                plot_dir=osp.join(eval_config.save_path, 'embed_dist'),
                plot_name='embed_dist_{}_{}'.format(layer, n),
                colors=['red', 'gray'],
            )

    torch.save((signals, noises, snrs), osp.join(config.save_path, f'embed_dist.pth'))