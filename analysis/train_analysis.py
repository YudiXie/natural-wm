import os
import os.path as osp

import pandas as pd

from analysis.plots import plot_train_log_loss, plot_train_log_acc

DIV_LINE_WIDTH = 50


def get_datasets(logdir):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.

    Return, list of tuple (model_name str, data pd.dataframe)
        for each model
    """
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            try:
                model_name = osp.split(root)[-1]
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
                datasets.append((model_name, exp_data))
            except:
                print('Imcomplete Data:', root)
    return datasets


def get_all_datasets(all_logdirs, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.

    Every entry in logdirs must end with '/' otherwise will be
        treated as prefix

    Return, list of tuple (model_name str, data pd.dataframe)
        for each model
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs by printing dir
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Load data from logdirs
    data = []
    for log in logdirs:
        data += get_datasets(log)
    return data


def plot_train_log(logdir, select=None, exclude=None,
                   plot_acc=True, exp_name=''):
    data = get_all_datasets(logdir, select, exclude)
    plot_train_log_loss(data, exp_name)
    if plot_acc:
        plot_train_log_acc(data, exp_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    args = parser.parse_args()

    plot_train_log(args.logdir, args.select, args.exclude)
