"""Delayed match to sample task"""
import os

import neurogym as ngym
import numpy as np
import torch.nn as nn
import torchvision
from gym import spaces

from configs.config_global import ROOT_DIR
from tasks.tasktools import ImageTrialEnv, display_env_trial, random


class DelayedMatch(ImageTrialEnv):
    def __init__(self, dt: int = 100, task_data: str = 'MNIST',
                 rewards: dict = None, timing: dict = None, **kwargs):
        super().__init__(dt=dt)
        # Possible decisions at the end of the trial
        self.choices = [1, 2]  # e.g. [match, non-match]

        # Optional rewards dictionary
        self.rewards = {'abort': -0.1, 'correct': +1.0, 'fail': 0.0}
        if rewards:
            self.rewards.update(rewards)

        # Optional timing dictionary
        # if provided, self.add_period can infer timing directly
        self.timing = {
            'sample': 500,
            'delay': 500,
            'test': 500}
        if timing:
            self.timing.update(timing)

        # define dataset
        self.task_data = task_data
        if self.task_data == 'MNIST':
            self.dataset = torchvision.datasets.MNIST(root=os.path.join(ROOT_DIR, 'data'),
                                                      train=True, download=True)
            # define observations_space
            self.datum_size = (28, 28, 1)
        elif self.task_data == 'CIFAR10':
            self.dataset = torchvision.datasets.CIFAR10(root=os.path.join(ROOT_DIR, 'data'),
                                                        train=True, download=True)
            # define observations_space
            self.datum_size = (32, 32, 3)
        else:
            raise NotImplementedError

        self.observation_space = spaces.Box(0, 255, shape=self.datum_size, dtype=np.uint8)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # define action_space
        # Optional annotation of the action space
        # Here we use ngym.spaces, which allows setting name of each dimension
        name = {'fixation': 0, 'choice': [1, 2]}
        self.action_space = ngym.spaces.Discrete(3, name=name)

    def _new_trial(self, **kwargs) -> dict:
        """
        self._new_trial() is called internally to generate a next trial.
        Typically, you need to
            set trial: a dictionary of trial information
            run self.add_period():
                will add time periods to the trial
                accessible through dict self.start_t and self.end_t
            run self.add_ob():
                will add observation to np array self.ob
            run self.set_groundtruth():
                will set groundtruth to np array self.gt
        Returns:
            trial: dictionary of trial information
        """
        # Setting trial information
        trial = {'ground_truth': random.choice(self.choices), }
        trial.update(kwargs)  # allows wrappers to modify the trial
        ground_truth = trial['ground_truth']

        # Adding periods sequentially
        self.add_period(['sample', 'delay', 'test'])

        # Setting observations
        id1, id2 = 0, 0
        while (id1 == id2):
            id1, id2 = random.randint(len(self.dataset)), random.randint(len(self.dataset))

        sample_img = np.array(self.dataset[id1][0])
        if len(sample_img.shape) == 2:
            sample_img = sample_img[:, :, None]
        delay_img = np.zeros(self.datum_size, dtype=np.uint8)

        self.set_ob(sample_img, 'sample')
        self.set_ob(delay_img, 'delay')
        self.set_mask(trial, 1, 'test')

        # match trials
        if ground_truth == 1:
            self.set_ob(sample_img, 'test')
        # non-match trials
        elif ground_truth == 2:
            test_img = np.array(self.dataset[id2][0])
            if len(test_img.shape) == 2:
                test_img = test_img[:, :, None]
            self.set_ob(test_img, 'test')
        else:
            raise NotImplementedError

        # Setting ground-truth value for supervised learning
        self.set_groundtruth(ground_truth, period='test')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # reward
        reward = 0
        gt = self.gt_now
        # observations
        if not self.in_period('test'):
            if action != 0:  # if fixation break
                reward = self.rewards['abort']
        else:
            if action != 0:
                new_trial = True
                if action == gt:  # if correct
                    reward = self.rewards['correct']
                    self.performance = 1
                else:  # if incorrect
                    reward = self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    # Instantiate the task
    env = DelayedMatch()
    display_env_trial(env)
