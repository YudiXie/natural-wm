import neurogym as ngym
import numpy as np
import os
from sympy import Dict, Idx
import torch
import torch.nn as nn
from gym import spaces

from datasets.continuousreport import angle2color
from tasks.changedetection import generate_pos
from tasks.tasktools import (ImageTrialEnv, display_env_trial,
                             sample_color_angles, random)
from utils.logger import Logger
from scipy.stats import circstd

class ContinuousReport(ImageTrialEnv):

    default_positions = [(13, 2), (4, 19), (22, 19), (4, 8), (13, 25), (22, 8), ]

    def __init__(self, dt: int = 100, rewards: dict = None, timing: dict = None,
                 num_patches: tuple = (3, ), 
                 output_uncertainty: bool = False, 
                 possible_colors: int = 360, 
                 fixed_positions = None, 
                 fixed_color_angles = None,
                 minimal_pairwise_distance = np.pi / 12,
                 img_size = (3, 32, 32),
                 patch_size = (5, 5),
                 num_position_configs = 1000,
                 **kwargs):
        super().__init__(dt=dt)
        # Optional rewards dictionary
        self.rewards = {'abort': -0.1, 'correct': +1.0, 'fail': 0.0}
        if rewards:
            self.rewards.update(rewards)

        # Optional timing dictionary
        # if provided, self.add_period can infer timing directly
        self.timing = {
            'sample': 500,
            'delay': 500,
            'report': 500}
        if timing:
            self.timing.update(timing)

        self.num_patches = num_patches
        self.possible_colors = possible_colors

        # define observation_space
        self.datum_size = (img_size[1], img_size[2], img_size[0])
        self.observation_space = spaces.Box(0, 255, shape=self.datum_size, dtype=np.uint8)
        self.fixed_positions = fixed_positions
        if fixed_positions == True:
            self.fixed_positions = self.default_positions

        self.fixed_color_angles = fixed_color_angles
        self.minimal_pairwise_distance = minimal_pairwise_distance
        self.patch_size = patch_size

        # can use compound action space
        #  to include a binary fixation action and continuous report action
        # define action_space
        # Optional annotation of the action space
        # Here we use ngym.spaces, which allows setting name of each dimension
        name = {'cos_angle': 0, 'sin_angle': 1}

        self.output_uncertainty = output_uncertainty
        self.action_space = ngym.spaces.Box(low=-1.0, high=1.0, shape=(2 + output_uncertainty, ),
                                        dtype=np.float32, name=name)

        self.position_configurations = {}
        for set_size in self.num_patches:
            pos_configs = [generate_pos(self.datum_size, self.patch_size, set_size) for _ in range(num_position_configs)]
            self.position_configurations[set_size] = pos_configs
        
    def criterion(self, output: torch.Tensor, target: torch.Tensor):
        mse = (output[:, :2] - target[:, :2]).square().mean(dim=-1)
        if not self.output_uncertainty:
            return mse
        sigma = output[:, -1].square()
        eps = 1e-2
        loss = mse / (sigma + eps) + torch.log(sigma)
        return loss

    @classmethod
    def get_image(cls, angles, positions=None):

        colors = []
        for angle in angles:
            if angle is not None:
                colors.append((angle2color(angle) * 255).astype(np.uint8))
            else:
                colors.append(np.zeros((3, ), dtype=np.uint8))
        num_patches = len(colors)
        
        if positions is None:
            positions = cls.default_positions[: num_patches]

        patches = [color[None, None, :] for color in colors]
        sample_img = np.full((32, 32, 3), 214, dtype=np.uint8)
        for pos, patch in zip(positions, patches):
            sample_img[pos[0]: pos[0] + 5, pos[1]: pos[1] + 5, :] = patch

        return sample_img

    def _set_up_stimuli(self, **kwargs) -> dict:

        # randomly decide the number of patches
        num_patches = random.choice(self.num_patches)

        # randomly select a cued object index
        cue_idx = random.randint(num_patches)
        trial = {'cue_idx': cue_idx, 'num_patches': num_patches}
        trial.update(kwargs)  # allows wrappers to modify the trial

        if self.fixed_positions is None:
            positions = random.choice(self.position_configurations[num_patches])
        else:
            positions = self.fixed_positions[: num_patches]

        # sample a list of random angle from the color wheel
        angles = sample_color_angles(
            num_patches, 
            possible_colors=self.possible_colors, 
            minimal_pairwise_distance=self.minimal_pairwise_distance
        )
        if self.fixed_color_angles is not None:
            for idx, angle in enumerate(self.fixed_color_angles):
                if angle is not None:
                    angles[idx] = angle

        colors = [(angle2color(angle) * 255).astype(np.uint8) for angle in angles]

        cued_color_angle = angles[cue_idx]
        trial.update({'positions': positions, 'color_angles': angles,
                      'cued_color_angle': cued_color_angle})

        # Setting observations
        # set sample image
        patches = [color[None, None, :] for color in colors]
        sample_img = np.full(self.datum_size, 214, dtype=np.uint8)
        for pos, patch in zip(positions, patches):
            sample_img[pos[0]: pos[0] + self.patch_size[0],
                       pos[1]: pos[1] + self.patch_size[1], :] = patch
        self.set_ob(sample_img, 'sample')

        # set delay image
        delay_img = np.full(self.datum_size, 214, dtype=np.uint8)
        self.set_ob(delay_img, 'delay')

        # set report image (right now, it is the same as cue image, no color wheel)
        cue_img = np.full(self.datum_size, 214, dtype=np.uint8)
        cued_pos = positions[cue_idx]
        cue_img[cued_pos[0]: cued_pos[0] + self.patch_size[0],
                cued_pos[1]: cued_pos[1] + self.patch_size[1], :] = 0
        self.set_ob(cue_img, 'report')

        # Setting ground-truth value for supervised learning
        # ground truth is the cos and sin of the cued color angle
        # The last element is added to guarantee that ground truth has 
        ground_truth = [np.cos(cued_color_angle), np.sin(cued_color_angle)] + self.output_uncertainty * [0, ]
        self.set_groundtruth(ground_truth, period='report')
        self.set_mask(trial, 1, period='report')

        return trial

    def _new_trial(self, **kwargs) -> dict:

        # Adding periods sequentially
        self.add_period(['sample', 'delay', 'report'])

        return self._set_up_stimuli(**kwargs)

    def _step(self, action):
        # here action should be continuous (cos_angle, sin_angle)
        new_trial = False
        # reward
        reward = 0
        gt = self.gt_now
        # observations
        if not self.in_period('report'):
            if max(abs(action)) >= 0.1:  # if fixation break
                reward = self.rewards['abort']
        else:
            if max(abs(action)) >= 0.1:
                new_trial = True
                if np.dot(action, gt) > 0.9:  # if correct
                    reward = self.rewards['correct']
                    self.performance = 1
                else:  # if incorrect
                    reward = self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}

    def before_test_callback(self, **unused):
        self.errors = {}
        self.uncertainty = {}
        for set_size in self.num_patches:
            self.errors[set_size] = []
            self.uncertainty[set_size] = []

    def test_step(self, labels, outputs, trial_info, **unused):
        for i, trial in enumerate(trial_info):
            pred = outputs[trial['trial_length'] - 1, i]
            pred_ang = torch.atan2(pred[1], pred[0]).item()

            delta = (pred_ang - trial['cued_color_angle']) / np.pi * 180
            delta -= (delta > 180) * 360
            delta += (delta < -180) * 360
            
            set_size = trial['num_patches']
            self.errors[set_size].append(delta)
            if self.output_uncertainty:
                self.uncertainty[set_size].append(pred[2].square().item())

    def after_test_callback(self, save_path, is_best, **unused):
        callback_results = {}
        for set_size in self.num_patches:
            error = self.errors[set_size]
            callback_results.update({f'error_std_{set_size}': circstd(error, low=-180, high=180)})
            if is_best:
                torch.save(error, os.path.join(save_path, f'error_{set_size}.pth'))

            if self.output_uncertainty:
                uncertainty = self.uncertainty[set_size]
                callback_results.update({f'log_sigma_{set_size}': np.log(uncertainty).mean()})
                if is_best:
                    torch.save(uncertainty, os.path.join(save_path, f'uncertainty_{set_size}.pth'))
        return callback_results


class CuedContinuousReport(ContinuousReport):

    def __init__(self, timing: dict = None, cue_prob: float = 1, **kwargs):
        super().__init__(**kwargs)

        self.timing = {
            'pro_cue': 500,
            'pro_cue_delay': 500,
            'sample': 500,
            'delay': 500,
            'report': 500
        }
        if timing:
            self.timing.update(timing)

        self.cue_prob = cue_prob

    def _new_trial(self, **kwargs) -> dict:

        # Adding periods sequentially
        self.add_period(['pro_cue', 'pro_cue_delay', 'sample', 'delay', 'report'])
        trial = self._set_up_stimuli(**kwargs)

        # with probability cue_prob, the prospetive cue is the same as the cue during the report phase
        idx = trial['cue_idx']
        if random.rand() > self.cue_prob:
            idx = random.randint(trial['num_patches'] - 1)
            if idx >= trial['cue_idx']:
                idx += 1
        trial['pro_cue_idx'] = idx

        # set delay image after the prospective cue
        delay_img = np.full(self.datum_size, 214, dtype=np.uint8)
        self.set_ob(delay_img, 'pro_cue_delay')

        # set the prospective cue
        cue_img = np.full(self.datum_size, 214, dtype=np.uint8)
        cued_pos = trial['positions'][idx]
        cue_img[cued_pos[0]: cued_pos[0] + self.patch_size[0],
                cued_pos[1]: cued_pos[1] + self.patch_size[1], :] = 0
        self.set_ob(cue_img, 'pro_cue')

        return trial


if __name__ == '__main__':
    # Instantiate the task
    env = ContinuousReport()
    display_env_trial(env)