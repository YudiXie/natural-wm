import neurogym as ngym
import numpy as np
import torch.nn as nn
import torch
import os
from gym import spaces

from tasks.changedetection import generate_pos
from datasets.continuousreport import angle2color
from tasks.continuousreport import ContinuousReport
from tasks.tasktools import (ImageTrialEnv, display_env_trial,
                             sample_color_angles, random)
from utils.logger import Logger
from scipy.stats import circstd


class ContinuousReportSeq(ContinuousReport):
    def __init__(self, dt: int = 100, rewards: dict = None, timing: dict = None,
                 num_patches: tuple = (3, ), **kwargs):
        super().__init__(dt=dt, num_patches=num_patches, rewards=rewards, **kwargs)

        # Optional timing dictionary
        # if provided, self.add_period can infer timing directly
        self.timing = {
            'delay': 200,
            'report': 500}
        for i in range(max(num_patches)):
            self.timing[f'sample{i}'] = 200
            self.timing[f'interval{i}'] = 100
        if timing:
            self.timing.update(timing)

    def _new_trial(self, **kwargs) -> dict:
        # randomly decide the number of patches
        num_patches = random.choice(self.num_patches)

        # Setting trial information
        # randomly select a cued object index
        cue_idx = random.randint(num_patches)
        trial = {'cue_idx': cue_idx, 'num_patches': num_patches}
        trial.update(kwargs)  # allows wrappers to modify the trial

        # Adding periods sequentially
        periods = []
        for i in range(num_patches):
            periods += [f'sample{i}', f'interval{i}']
        periods += ['delay', 'report']
        self.add_period(periods)

        positions = random.choice(self.position_configurations[num_patches])
        # sample a list of random angle from the color wheel
        angles = sample_color_angles(num_patches)
        colors = [(angle2color(angle) * 255).astype(np.uint8) for angle in angles]
        patches = [color[None, None, :] for color in colors]

        cued_color_angle = angles[cue_idx]
        trial.update({'positions': positions, 'color_angles': angles,
                      'cued_color_angle': cued_color_angle})

        # Setting observations
        # set sample images
        for i, pos in enumerate(positions):
            sample_img = np.full(self.datum_size, 214, dtype=np.uint8)
            sample_img[pos[0]: pos[0] + self.patch_size[0],
                       pos[1]: pos[1] + self.patch_size[1], :] = patches[i]
            self.set_ob(sample_img, f'sample{i}')

        # set delay image
        delay_img = np.full(self.datum_size, 214, dtype=np.uint8)
        self.set_ob(delay_img, 'delay')
        for i in range(num_patches):
            self.set_ob(delay_img, f'interval{i}')

        # set report image (right now, it is the same as cue image, no color wheel)
        cue_img = np.full(self.datum_size, 214, dtype=np.uint8)
        cued_pos = positions[cue_idx]
        cue_img[cued_pos[0]: cued_pos[0] + self.patch_size[0],
                cued_pos[1]: cued_pos[1] + self.patch_size[1], :] = 0
        self.set_ob(cue_img, 'report')

        # Setting ground-truth value for supervised learning
        # ground truth is the cos and sin of the cued color angle
        ground_truth = np.cos(cued_color_angle), np.sin(cued_color_angle)
        self.set_groundtruth(ground_truth, period='report')
        self.set_mask(trial, 1, period='report')

        return trial

    def before_test_callback(self, **unused):
        self.errors = {}
        self.detail_errors = {}
        for set_size in self.num_patches:
            self.errors[set_size] = []
            self.detail_errors[set_size] = []
            for i in range(set_size):
                self.detail_errors[set_size].append([])

    def test_step(self, labels, outputs, trial_info, **unused):
        for i, trial in enumerate(trial_info):
            pred = outputs[trial['trial_length'] - 1, i]
            pred_ang = torch.atan2(pred[1], pred[0]).item()

            delta = (pred_ang - trial['cued_color_angle']) / np.pi * 180
            delta -= (delta > 180) * 360
            delta += (delta < -180) * 360
            
            set_size = trial['num_patches']
            self.errors[set_size].append(delta)
            self.detail_errors[set_size][trial['cue_idx']].append(delta)

    def after_test_callback(self, save_path, is_best, **unused):
        callback_results = {}
        for set_size in self.num_patches:
            error = self.errors[set_size]
            callback_results.update({f'error_std_{set_size}': circstd(error, low=-180, high=180)})

            if is_best:
                torch.save(error, os.path.join(save_path, f'error_{set_size}.pth'))

        for set_size in self.num_patches:
            for rank in range(set_size):
                errors = self.detail_errors[set_size][rank]
                callback_results.update({f'error_std_{set_size}_{rank}': circstd(errors, low=-180, high=180)})
        return callback_results
        

if __name__ == '__main__':
    # Instantiate the task
    env = ContinuousReportSeq(num_patches=5)
    display_env_trial(env)
