from configs.configs import BaseConfig
from eval import model_eval
from tasks.tasktools import ImageTrialEnv_Wrapper
from tasks.continuousreport import ContinuousReport
from copy import deepcopy
import torch

def get_scores_and_targets(config: BaseConfig, test_batch=100):
    """
    return scores (logits) and targets for a given model, for each set size
    """
    eval_config = deepcopy(config)
    eval_config.test_batch = test_batch
    eval_config.num_workers = 1
    
    class Eval_Wrapper(ImageTrialEnv_Wrapper):

        def before_test_callback(self, **kwargs):
            self.env: ContinuousReport
            self.env.before_test_callback(**kwargs)
            self.scores = [[] for _ in self.env.num_patches]
            self.targets = [[] for _ in self.env.num_patches]

        def test_step(self, labels, outputs, trial_info, **kwargs):
            self.env.test_step(labels=labels, outputs=outputs, trial_info=trial_info, **kwargs)
            for i, trial in enumerate(trial_info):
                score = torch.softmax(outputs[trial['trial_length'] - 1, i], dim=0)[1].item()
                # score is the confidence of no change
                label = 2 - labels[trial['trial_length'] - 1, i].item()
                # no change 1, change 0
                for idx, set_size in enumerate(self.env.num_patches):
                    if trial['num_patches'] == set_size:
                        self.scores[idx].append(score)
                        self.targets[idx].append(label)
                        break

        def after_test_callback(self, **kwargs):
            self.env.after_test_callback(**kwargs)
            return self.scores, self.targets

    return model_eval(eval_config, env_wrapper=Eval_Wrapper)


def get_hit_and_false_alarm_rates(scores, targets, threshold):
    """
    Get hit rate and false alarm rate for a given threshold.
    targets: 1d array of 0s and 1s
    scores: 1d array of scores, in range (0, 1)
    threshold: float in range (0, 1)
    """
    tp = ((targets == 1) & (scores > threshold)).sum()
    fn = ((targets == 1) & (scores <= threshold)).sum()
    fp = ((targets == 0) & (scores > threshold)).sum()
    tn = ((targets == 0) & (scores <= threshold)).sum()
    
    tpr = tp / (tp + fn)  # hit rate
    fpr = fp / (fp + tn)  # false alarm rate
    return tpr, fpr
