import os
import pandas as pd
import configs.experiments as experiments


def check_run_complete(cfg) -> bool:
    """
    Check if the run is complete by checking both the complete
    file and the progress file
    :param cfg: config
    :return: True if train is complete and progress file is valid, False otherwise
    """

    exp_str = cfg.save_path.split('/')[-1]
    progress_path = os.path.join(cfg.save_path, 'progress.txt')
    complete_path = os.path.join(cfg.save_path, 'train_complete.txt')
    run_complete = True

    if not os.path.exists(complete_path):
        print('No complete record for: ' + exp_str)
        run_complete = False
    else:
        try:
            exp_data = pd.read_table(progress_path)
            if len(exp_data) < cfg.max_batch // cfg.log_every:
                print('Training progress is not complete for: ' + exp_str)
                run_complete = False
        except:
            print('No data for: ' + exp_str)
            run_complete = False
    
    return run_complete


def get_missing_runs(config_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if there are any missing runs in the experiments
    :param config_df: dataframe of experiments
    :return: mis_config_df, dataframe of missing runs
    """
    run_n_cmplt = [not check_run_complete(cfg) for cfg in config_df.loc[:, 'config']]
    if all([not val for val in run_n_cmplt]):
        print('All runs completed!')
    return config_df[run_n_cmplt]


if __name__ == '__main__':
    cfg_df = experiments.classification_pretrain_new()
    print(get_missing_runs(cfg_df))
