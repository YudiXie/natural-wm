import os
from collections import defaultdict
import pickle
import numpy as np
import torch
from configs.configs import BaseConfig
from datasets.data_sets import init_single_dataset

from utils.train_utils import model_init, task_init, env_dict
from tasks.taskfunctions import data_batch_to_device
from tasks.tasktools import ImageTrialEnv_Wrapper
from copy import deepcopy
from eval import model_eval


def append_tuple(name, activity_dict, output):
    for i_, otp in enumerate(output):
        new_name = name + '_' + str(i_ + 1)
        if isinstance(otp, tuple):
            append_tuple(new_name, activity_dict, otp)
        elif isinstance(otp, torch.Tensor):
            activity_dict[new_name].append(otp.detach().cpu().numpy())
        else:
            raise NotImplementedError('append type not implemented')


def append_activations(name, activity_dict):
    """
    Returns a hook function that can be registered with model layers
        to obtain and store the output history of hidden activations in activation_dict
    name: the name of module to record activities
    activity_dict: a collection.defaultdict with default factory function set to list
    """
    assert isinstance(activity_dict, defaultdict) \
           and activity_dict.default_factory == list, 'activity_dict must be default dict'

    def hook(module, inp, otp):
        if isinstance(otp, torch.Tensor):
            activity_dict[name].append(otp.detach().cpu().numpy())
        elif isinstance(otp, tuple):
            append_tuple(name, activity_dict, otp)
        else:
            raise NotImplementedError('append type not implemented')

    return hook


def record_eval_activity(config, layers=['rnn'], eval_iter=100):
    config.Batch_Size = 4
    task_func = task_init(config)
    task_batch_s = task_func.task_batch_s

    test_loader, datum_size = init_single_dataset(config.dataset, False, task_batch_s,
                                                  config.diff_loc, config.inc_targets, config)
    net = model_init(config, [datum_size], 'eval')

    # register forward hook for all layers to record activity from
    batch_activity = defaultdict(list)
    all_activity = defaultdict(list)
    label_info = []
    correct_info = []

    for name, m in net.named_modules():
        if layers is None or name in layers:
            m.register_forward_hook(append_activations(name, batch_activity))

    correct = 0
    total = 0

    test_b = 0
    for test_data in test_loader:
        test_b += 1
        if test_b > eval_iter:
            break
        with torch.no_grad():
            output, num, num_corr = task_func.roll(net, test_data, evaluate=True)
            input_, label_ = data_batch_to_device(test_data)[:2]
        output = output.int()
        total += num
        correct += num_corr

        for k_, v_ in batch_activity.items():
            if k_ not in ['rnn_2', 'rnn_3']:
                # all_activity after stack (t_steps, batch_size, neurons)
                # only record hidden activity 
                if k_ == 'rnn_1':
                    all_activity['rnn'].append(np.stack(v_, axis=0)) 
                else:  
                    all_activity[k_].append(np.stack(v_, axis=0))
            batch_activity[k_] = []
        if 'input' in layers:
            all_activity['input'].append(np.stack(input_.detach().cpu().numpy().reshape(1, task_batch_s, -1), axis=0))

        # record labels for each activity
        label_info.append(np.stack(label_.detach().cpu().numpy(), axis=0))
        correct_info.append(np.stack(output.detach().cpu().numpy(), axis=0))

    for k_, datum in all_activity.items():
        all_activity[k_] = np.stack(datum)
    
    net_acc = (correct / total) * 100.0
    print("net accuracy:", net_acc)
    
    for k_, datum in all_activity.items():
        activity_save_path = os.path.join(config.save_path, 'raw_act_' + config.dataset + config.task_type + k_ + '.pkl')
        with open(activity_save_path, 'wb') as save_file:
            pickle.dump({k_: datum}, save_file)
    np.save(os.path.join(config.save_path, 'label_info_' + config.dataset + config.task_type + '.npy'), label_info)
    np.save(os.path.join(config.save_path, 'correct_info_' + config.dataset + config.task_type + '.npy'), correct_info)        
    
    print('successfully saved activity:\n{}\n'.format(activity_save_path))

def record_activity_ngym(
    config: BaseConfig,
    layers,
    test_batch=100,
    store_activity=None,
    load_saved=False
):
    """
    Record neural activity (with neurogym envs)
    :param layers: a list of layers to record, e.g. layers=['cnn', 'rnn', 'resblock1']
    :param eval_iter: number of batches to record
    :param store_activity: 
        an customized function f(stored_activity, recorded_activity, layer_name, trial_info), where
            :param stored_activity: a dictionary of lists where the keys are name of layers
            :param recorded_activity: a numpy array of shape (trial_length, *layer_shape)
            :param layername: e.g., 'cnn', 'rnn'
            :param trial_info: a dictionary returned by env.new_trial()
        Typically, the customized function should append infomation to stored_activity[layer_name]
    :return: a tuple (activity, batch_info).
    :return trial_info: list of trial_info
    """
    # check if activity is already recorded
    if load_saved and os.path.exists(os.path.join(config.save_path, 'activity.pth')):
        print('activity already recorded, loading from saved file')
        all_activity, trial_info = torch.load(os.path.join(config.save_path, 'activity.pth'))
        return all_activity, trial_info

    eval_config = deepcopy(config)
    eval_config.test_batch = test_batch
    eval_config.num_workers = 0

    # register forward hook for all layers to record activity from
    batch_activity = defaultdict(list)
    all_activity = defaultdict(list)
    handles = defaultdict(list)
    all_trial_info = []

    class Eval_Wrapper(ImageTrialEnv_Wrapper):

        def before_test_callback(self, model, **kwargs):
            self.env.before_test_callback(model=model, **kwargs)
            for layer in layers:
                if layer == 'resblock1':
                    handles[layer] = model.cnn.cnn.layer1.register_forward_hook(
                        append_activations(layer, batch_activity)
                    )
                elif layer == 'resblock2':
                    handles[layer] = model.cnn.cnn.layer2.register_forward_hook(
                        append_activations(layer, batch_activity)
                    )
                elif layer == 'resblock3':
                    handles[layer] = model.cnn.cnn.layer3.register_forward_hook(
                        append_activations(layer, batch_activity)
                    )
                else:
                    handles[layer] = model.__getattr__(layer).register_forward_hook(
                        append_activations(layer, batch_activity)
                    )

        def test_step(self, trial_info, **kwargs):
            self.env.test_step(trial_info=trial_info, **kwargs)
            for k_, v_ in batch_activity.items():
                act = np.stack(v_, axis=1)
                bsz = act.shape[0]
                for idx in range(bsz):
                    if store_activity is not None:
                        store_activity(all_activity, act[idx], k_, trial_info[idx])
                    else:
                        all_activity[k_].append(act[idx])
                batch_activity[k_] = []
            all_trial_info.extend(trial_info)            

        def after_test_callback(self, save_path, **kwargs):
            self.env.after_test_callback(save_path=save_path, **kwargs)
            for layer in layers:
                handles[layer].remove()

            # save activity
            torch.save((all_activity, all_trial_info), os.path.join(save_path, 'activity.pth'))

    model_eval(eval_config, env_wrapper=Eval_Wrapper)
    return all_activity, all_trial_info