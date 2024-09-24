import sys
import argparse
import logging
import os
import subprocess

import configs.exp_analysis as exp_analysis
import configs.exp_eval as exp_eval
import configs.experiments as experiments
from configs.config_global import LOG_LEVEL, ANA_LOG_LEVEL, ROOT_DIR
from configs.configs import BaseConfig
from eval import model_eval
from utils.config_utils import configs_df_unpack, save_config
from utils.general_utils import get_missing_runs


def train_cmd(save_path, use_neurogym):
    arg = '\'' + save_path + '\''
    if use_neurogym:
        command = f'python -c "import train; train.train_from_path({arg})"'
    else:
        command = f'python -c "import train_old; train_old.train_from_path({arg})"'
    return command


def eval_cmd(eval_save_path):
    arg = '\'' + eval_save_path + '\''
    command = f'python -c "import eval; eval.model_eval({arg})"'
    return command


def analysis_cmd(exp_name):
    arg = exp_name + '_analysis()'
    command = f'python -c "import configs.exp_analysis as exp_analysis; exp_analysis.{arg}"'
    return command


def get_jobfile(cmd, 
                job_name, 
                dep_ids, 
                email=False,
                sbatch_path='./sbatch/', 
                hours=8, 
                partition=['normal'], 
                mem=32, 
                gpu_constraint='high-capacity', 
                cpu=2,
                output_path='./sbatch/'):
    """
    Create a job file.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        cmd: python command to be execute by the cluster
        job_name: str, name of the job file
        dep_ids: list, a list of job ids used for job dependency
        email: bool, whether or to send email about job status
        sbatch_path : str, Directory to store SBATCH
        hours : int, number of hours to train
        output_path : str, Directory to store output
        partition : list, a list of cluster partition to use
    Returns:
        job_file : str, Path to the job file.
    """
    assert type(dep_ids) is list, 'dependency ids must be list'
    assert all(type(id_) is str for id_ in dep_ids), 'dependency ids must all be strings'

    work_directory = ROOT_DIR

    if len(dep_ids) == 0:
        dependency_line = ''
    else:
        dependency_line = '#SBATCH --dependency=afterok:' \
                          + ':'.join(dep_ids) + '\n'

    if not email:
        email_line = ''
    else:
        email_line = '#SBATCH --mail-type=ALL\n' + \
                     '#SBATCH --mail-user=rxie9596@outlook.com\n'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(sbatch_path):
        os.makedirs(sbatch_path)
    job_file = os.path.join(sbatch_path, job_name + '.sh')

    with open(job_file, 'w') as f:
        f.write(
            '#!/bin/bash\n'
            + '#SBATCH -t {}:00:00\n'.format(hours)
            + '#SBATCH -N 1\n'
            + '#SBATCH -n {}\n'.format(cpu)
            + '#SBATCH --mem={}G\n'.format(mem)
            + '#SBATCH --gres=gpu:1\n'
            + '#SBATCH --constraint={}\n'.format(gpu_constraint)
            + '#SBATCH --partition={}\n'.format(','.join(partition))
            + '#SBATCH -e {}/slurm-%j-{}.out\n'.format(output_path, job_name)
            + '#SBATCH -o {}/slurm-%j-{}.out\n'.format(output_path, job_name)
            + dependency_line
            + email_line
            + '\n'
            + 'source ~/.bashrc\n'
            + 'module load openmind8/cuda/11.7\n'
            + 'conda activate wm\n'
            + 'cd ' + work_directory + '\n'
            + 'echo -e "System Info: \\n----------\\n$(hostnamectl)\\n----------"' + '\n'
            + 'nvcc --version\n'
            + 'nvidia-smi\n'
            + cmd + '\n'
            + '\n'
            )
        print(job_file)
    return job_file


def train_experiment(experiment: str,
                     on_cluster: bool,
                     use_exp_array: bool,
                     partition: str,
                     user: str,
                     run_missing: bool):
    """Train model across platforms given experiment name.
    adapted from https://github.com/gyyang/olfaction_evolution
    Args:
        experiment: str, name of experiment to be run
            must correspond to a function in experiments.py
        on_cluster: bool, whether to run experiments on cluster
        use_exp_array: use dependency between training of different exps
        partition: list, a list of cluster partition to use
        user: str, user name
        run_missing: bool, if True to run missing experiments,
          otherwise run the whole experiment

    Returns:
        return_ids: list, a list of job ids that are last in the dependency sequence
            if not using cluster, return is an empty list
    """
    print('Training {:s} experiment'.format(experiment))
    if experiment in dir(experiments):
        # Get list of configurations from experiment function
        exp_configs = getattr(experiments, experiment)()
    else:
        raise ValueError('Experiment config not found: ', experiment)

    return_ids = []
    if not use_exp_array:
        # exp_configs is a config_df
        if run_missing:
            exp_configs = get_missing_runs(exp_configs)
            if input('Continue to submit? (yes/no): ') != 'yes':
                sys.exit("exit program.")

        exp_configs = configs_df_unpack(exp_configs)
        if len(exp_configs) != 0:
            assert isinstance(exp_configs[0], BaseConfig), \
                'exp_configs should be list of configs'

        if on_cluster:
            for config in exp_configs:
                if not save_config(config, config.save_path):
                    continue
                python_cmd = train_cmd(config.save_path, config.neurogym)
                job_n = config.experiment_name + '_' + config.model_name
                output_path = os.path.join(os.path.dirname(config.save_path), 'slurm_output')
                cp_process = subprocess.run(['sbatch', get_jobfile(python_cmd,
                                                                   job_n,
                                                                   dep_ids=[],
                                                                   hours=config.hours, 
                                                                   partition=partition,
                                                                   mem=config.mem,
                                                                   gpu_constraint=config.gpu_constraint,
                                                                   cpu=config.cpu,
                                                                   sbatch_path=config.save_path,
                                                                   output_path=output_path
                                                                   )],
                                            capture_output=True, check=True)
                cp_stdout = cp_process.stdout.decode()
                print(cp_stdout)
                job_id = cp_stdout[-9:-1]
                return_ids.append(job_id)
        else:
            for config in exp_configs:
                if save_config(config, config.save_path):
                    if config.neurogym:
                        from train import model_train
                    else:
                        from train_old import model_train
                    model_train(config)
    else:
        # exp_configs is a list of config_df
        if run_missing:
            exp_configs = [get_missing_runs(cfg_df) for cfg_df in exp_configs]
            if input('Continue to submit? (yes/no): ') != 'yes':
                sys.exit("exit program.")

        exp_configs = [configs_df_unpack(cfg_df) for cfg_df in exp_configs]
        
        if len(exp_configs) != 0:
            if len(exp_configs[0]) != 0:
                assert isinstance(exp_configs[0], list) \
                    and isinstance(exp_configs[0][0], BaseConfig), \
                    'exp_configs should a list of lists of configs'

        if on_cluster:
            send_email = False
            pre_job_ids = []
            for group_num, config_group in enumerate(exp_configs):
                group_job_ids = []
                for config in config_group:
                    if not save_config(config, config.save_path):
                        continue

                    if group_num == len(exp_configs) - 1:
                        send_email = True

                    python_cmd = train_cmd(config.save_path, config.neurogym)
                    job_n = config.experiment_name + '_' + config.model_name
                    cp_process = subprocess.run(['sbatch',
                                                 get_jobfile(python_cmd, job_n,
                                                             dep_ids=pre_job_ids,
                                                             email=send_email,
                                                             hours=config.hours, 
                                                             partition=partition,)], # TODO: Update this line
                                                capture_output=True, check=True)
                    cp_stdout = cp_process.stdout.decode()
                    print(cp_stdout)
                    job_id = cp_stdout[-9:-1]
                    group_job_ids.append(job_id)
                pre_job_ids = group_job_ids

            return_ids = pre_job_ids

        else:
            for config_group in exp_configs:
                for config in config_group:
                    if save_config(config, config.save_path):
                        if config.neurogym:
                            from train import model_train
                        else:
                            from train_old import model_train
                        model_train(config)

    return return_ids


def eval_experiment(experiment, prev_ids, on_cluster, partition, user):
    """evaluate experiment
     Args:
         experiment: str, name of experiment to be evaluated
             must correspond to a function in exp_eval.py
         prev_ids: list, list of previous job ids
         on_cluster: bool, if use on the cluster

    Returns:
        return_ids: list, a list of job ids that are last in the dependency sequence
            if not using cluster, return is an empty list
     """
    print('Evaluating {:s} experiment'.format(experiment))
    if (experiment + '_eval') in dir(exp_eval):
        eval_config_list = getattr(exp_eval, experiment + '_eval')()
    else:
        raise ValueError('Experiment evaluation not found: ', experiment + '_eval')

    return_ids = []
    for config in eval_config_list:
        if on_cluster:
            eval_save_path = os.path.join(config.save_path, 'eval_configs',
                                          'net' + str(config.eval_net_num),
                                          config.eval_name)
            save_config(config, eval_save_path)
            python_cmd = eval_cmd(eval_save_path)
            job_n = config.experiment_name + '_' + config.model_name \
                    + '_eval_net_' + str(config.eval_net_num) + config.eval_name
            slurm_cmd = ['sbatch', get_jobfile(python_cmd, job_n,
                                               dep_ids=prev_ids, email=True,
                                               partition=partition)] # TODO: Update this line
            cp_process = subprocess.run(slurm_cmd, capture_output=True,
                                        check=True)
            cp_stdout = cp_process.stdout.decode()
            print(cp_stdout)
            job_id = cp_stdout[-9:-1] 
            return_ids.append(job_id)
        else:
            model_eval(config)

    return return_ids

def analyze_experiment(experiment, prev_ids, on_cluster, partition, user):
    """analyze experiments
     adapted from https://github.com/gyyang/olfaction_evolution

     Args:
         experiment: str, name of experiment to be analyzed
             must correspond to a function in exp_analysis.py
         prev_ids: list, list of previous job ids
         on_cluster: bool, if use on the cluster
     """
    print('Analyzing {:s} experiment'.format(experiment))
    if (experiment + '_analysis') in dir(exp_analysis):
        if on_cluster:
            python_cmd = analysis_cmd(experiment)
            job_n = experiment + '_analysis'
            slurm_cmd = ['sbatch', get_jobfile(python_cmd, job_n,
                                               dep_ids=prev_ids, email=False, partition=partition,
                                               hours=24, mem=64,)] # TODO: Update this line
            cp_process = subprocess.run(slurm_cmd, capture_output=True,
                                        check=True)
            cp_stdout = cp_process.stdout.decode()
            print(cp_stdout)
        else:
            getattr(exp_analysis, experiment + '_analysis')()
    else:
        raise ValueError('Experiment analysis not found: ', experiment + '_analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs='+', help='Train experiments', default=[])
    parser.add_argument('-e', '--evaluate', nargs='+', help='Evaluate models', default=[])
    parser.add_argument('-a', '--analyze', nargs='+', help='Analyze experiments', default=[])

    parser.add_argument('-c', '--cluster', action='store_true', help='Use batch submission on cluster')
    parser.add_argument('-m', '--missing', action='store_true', help='Run missing experiments')

    parser.add_argument('-p', '--partition', nargs='+', default=['normal'], help='Partition of resource on cluster to use')
    parser.add_argument('-u', '--user', default='yu_xie', help='User name on cluster')
    
    args = parser.parse_args()
    experiments2train = args.train
    experiments2eval = args.evaluate
    experiments2analyze = args.analyze
    use_cluster = args.cluster
    # on openmind cluster
    # use_cluster = 'node' in platform.node() or 'dgx' in platform.node()

    if len(args.partition) > 1:
        raise NotImplementedError('Only one partition is supported on openmind at the moment')

    # evaluation jobs are executed after training jobs,
    # analysis jobs are executed after training and evaluation jobs.
    train_ids = []
    if experiments2train:
        logging.basicConfig(level=LOG_LEVEL)
        for exp in experiments2train:
            exp_array = '_exp_array' in exp
            exp_ids = train_experiment(exp, on_cluster=use_cluster,
                                       use_exp_array=exp_array, 
                                       partition=args.partition,
                                       user=args.user, 
                                       run_missing=args.missing)
            train_ids += exp_ids

    eval_ids = []
    if experiments2eval:
        logging.basicConfig(level=LOG_LEVEL)
        for exp in experiments2eval:
            exp_ids = eval_experiment(exp, prev_ids=train_ids,
                                      on_cluster=use_cluster, 
                                      partition=args.partition,
                                      user=args.user)
            eval_ids += exp_ids

    if experiments2analyze:
        logging.basicConfig(level=ANA_LOG_LEVEL)
        for exp in experiments2analyze:
            analyze_experiment(exp, prev_ids=train_ids + eval_ids,
                               on_cluster=use_cluster, 
                               partition=args.partition,
                               user=args.user)
