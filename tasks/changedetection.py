"""Change detection task"""
import neurogym as ngym
import numpy as np
import torch
import torch.nn as nn
from gym import spaces

from datasets.luckvogel import generate_pos
from tasks.dms import DelayedMatch
from tasks.tasktools import (angle2color, display_env_trial, get_changed_angle,
                             get_opposite_angle, random, sample_color_angles)
from utils.logger import Logger

# gray
lg_gray_color = np.array([214, 214, 214], dtype=np.uint8)
# red, blue, green, purple, black
lg_colors = np.array([[209, 55, 49],
                      [36, 88, 159],
                      [97, 169, 76],
                      [177, 49, 126],
                      [0, 0, 0]], dtype=np.uint8)

def check_overlap(pos1, pos2, p_size):
    x1, y1 = pos1
    x2, y2 = pos2
    if x2 - p_size[0] <= x1 <= x2 + p_size[0] \
            and y2 - p_size[1] <= y1 <= y2 + p_size[1]:
        return True
    else:
        return False

buffers = []

def generate_pos(image_size, patch_size, n_patch):
    """
    Generate positions of patches
    args:
        image_size: (height, width, channels) 3-dimensional image size
        patch_size: (height, width) two-dimensional patch size
        n_patch: number of patches in the array
    returns:
        pos_array: array of positions
    """
    pos_array = []
    attempts = 0
    while len(pos_array) < n_patch:
        attempts += 1

        if attempts > 100:
            # print("attempts over 100, empty array can start over")
            pos_array = []
            attempts = 0

        x = random.randint(image_size[0] - patch_size[0])
        y = random.randint(image_size[1] - patch_size[1])
        new_pos = (x, y)

        # check if there are overlaps
        flag = True
        for prev_pos in pos_array:
            if check_overlap(new_pos, prev_pos, patch_size):
                flag = False
                break

        if flag:
            pos_array.append(new_pos)

    return pos_array

def paint(positions, color_angles, img_size, lg_patch_size = (5, 5)):
    """
    Paint the image with patches based on positions and attributes
    args:
        positions: array of positions
        color_angles: list of color angles, each in range [-pi, pi)
        img_size: (height, width, channels) 3-dimensional image size
    returns:
        image: np.ndarray, image with patches,
            PIL image format (H x W x C) in the range [0, 255]
    """
    image = lg_gray_color[0] * np.ones(img_size, dtype=np.uint8)
    for pos, angle in zip(positions, color_angles):
        patch = (angle2color(angle) * 255).astype(np.uint8)[None, None, :]
        image[pos[0]: pos[0] + lg_patch_size[0],
              pos[1]: pos[1] + lg_patch_size[1], :] = patch
    return image

def paint_discrete_colors(positions, color_ids, img_size, lg_patch_size = (5, 5)):
    """
    Paint the image with patches based on positions and attributes
    args:
        positions: array of positions
        color_ids: list of color ids, each in range [0, 4]
        img_size: (height, width, channels) 3-dimensional image size
    returns:
        image: np.ndarray, image with patches,
            PIL image format (H x W x C) in the range [0, 255]
    """
    image = lg_gray_color[0] * np.ones(img_size, dtype=np.uint8)
    for pos, color_id in zip(positions, color_ids):
        patch = lg_colors[color_id][None, None, :]
        image[pos[0]: pos[0] + lg_patch_size[0],
              pos[1]: pos[1] + lg_patch_size[1], :] = patch
    return image

class ChangeDetection(DelayedMatch):

    def __init__(self, dt: int = 100, 
                 rewards: dict = None, 
                 timing: dict = None,
                 num_patches: tuple = (3, ), 
                 easy_mode = False, 
                 change_magnitude = None, 
                 use_fixed_colors: bool = False, 
                 img_size = (3, 32, 32), 
                 lg_patch_size = (5, 5), 
                 num_position_configs: int = 1000, 
                 **kwargs):
        super(DelayedMatch, self).__init__(dt=dt)
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

        # parameters for the data
        self.num_patches = num_patches
        self.datum_size = (img_size[1], img_size[2], img_size[0])
        self.observation_space = spaces.Box(0, 255, shape=self.datum_size, dtype=np.uint8)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.use_fixed_colors = use_fixed_colors
        self.lg_patch_size = lg_patch_size

        self.easy_mode = easy_mode
        self.change_magnitude = change_magnitude

        # define action_space
        # Optional annotation of the action space
        # Here we use ngym.spaces, which allows setting name of each dimension
        name = {'fixation': 0, 'choice': [1, 2]}
        self.action_space = ngym.spaces.Discrete(3, name=name)
        
        self.position_configurations = {}
        for set_size in self.num_patches:
            pos_configs = [generate_pos(self.datum_size, self.lg_patch_size, set_size) for _ in range(num_position_configs)]
            self.position_configurations[set_size] = pos_configs

    def _new_trial(self, **kwargs) -> dict:
        # Setting trial information
        trial = {'ground_truth': random.choice(self.choices)}
        trial.update(kwargs)  # allows wrappers to modify the trial
        ground_truth = trial['ground_truth']

        # Adding periods sequentially
        self.add_period(['sample', 'delay', 'test'])

        # randomly decide the number of patches
        num_patches = random.choice(self.num_patches)
        trial['num_patches'] = num_patches

        # sample positions of patches
        positions = random.choice(self.position_configurations[num_patches])

        # sample color angles that are different from each other
        if self.use_fixed_colors:
            original_color_ids = [random.randint(len(lg_colors)) for _ in range(num_patches)]
        else:
            original_angles = sample_color_angles(num_patches, minimal_pairwise_distance=0)
        
        # default using dissimilar sample mode,
        # TODO: this might cause problems because the changed color could be similar to one of the distractors
        # change the color of the last random sample to the opposite color on a color wheel

        if not self.easy_mode:
            if self.use_fixed_colors:
                changed_color_ids = original_color_ids.copy()
                changed_color_ids[-1] = (original_color_ids[-1] + 2) % len(lg_colors)
            else:
                changed_angles = original_angles.copy()
                if self.change_magnitude is None:
                    changed_angles[-1] = random.rand() * 2 * np.pi - np.pi
                else:
                    changed_angles[-1] = get_changed_angle(original_angles[-1], self.change_magnitude)
            changed_positions = positions
        else:
            # in the easy mode, colors and positions of all patches will change
            changed_num_patches = random.choice(self.num_patches)
            if self.use_fixed_colors:
                changed_color_ids = [random.randint(len(lg_colors)) for _ in range(num_patches)]
            else:
                changed_angles = sample_color_angles(num_patches, minimal_pairwise_distance=0)
            changed_positions = random.choice(self.position_configurations[changed_num_patches])

        trial.update({'positions': positions, 'changed_positions': changed_positions})
        if self.use_fixed_colors:
            trial.update({'original_color_ids': original_color_ids, 'changed_color_ids': original_color_ids})
            sample_img = paint_discrete_colors(positions, original_color_ids, self.datum_size, self.lg_patch_size)
        else:
            trial.update({'original_angles': original_angles, 'changed_angles': original_angles})
            sample_img = paint(positions, original_angles, self.datum_size, self.lg_patch_size)
        
        delay_img = lg_gray_color[0] * np.ones(self.datum_size, dtype=np.uint8)

        # Setting observations
        self.set_ob(sample_img, 'sample')
        self.set_ob(delay_img, 'delay')
        self.set_mask(trial, 1, 'test')

        # match trials
        if ground_truth == 1:
            self.set_ob(sample_img, 'test')
            # no change, so no need to update trial
        # non-match trials
        elif ground_truth == 2:
            if self.use_fixed_colors:
                test_img = paint_discrete_colors(changed_positions, changed_color_ids, self.datum_size, self.lg_patch_size)
                trial.update({'changed_color_ids': changed_color_ids})
            else:
                test_img = paint(changed_positions, changed_angles, self.datum_size, self.lg_patch_size)
                trial.update({'changed_angles': changed_angles})  # change
            self.set_ob(test_img, 'test')
        else:
            raise NotImplementedError

        # Setting ground-truth value for supervised learning
        self.set_groundtruth(ground_truth, period='test')

        return trial
    
    def before_test_callback(self, **unused):
        self.pred_num = {}
        self.pred_correct = {}
        self.TP = {}
        self.FP = {}
        self.TN = {}
        self.FN = {}
        for set_size in self.num_patches:
            self.pred_num[set_size] = self.pred_correct[set_size] = 0
            self.TP[set_size] = self.FP[set_size] = self.TN[set_size] = self.FN[set_size] = 0

    def test_step(self, labels, outputs, loss_mask, trial_info, **unused):
        for i, trial in enumerate(trial_info):
            logits = outputs[trial['trial_length'] - 1, i]
            probs = torch.softmax(logits, dim=-1)
            confidence, prediction = probs.max(dim=0)
            label = labels[trial['trial_length'] - 1, i]
            
            set_size = trial['num_patches']
            self.pred_num[set_size] += 1
            self.pred_correct[set_size] += (prediction == label).item()

            self.TP[set_size] += ((prediction == 2) & (label == 2)).item()
            self.FP[set_size] += ((prediction == 2) & (label == 1)).item()
            self.TN[set_size] += ((prediction == 1) & (label == 1)).item()
            self.FN[set_size] += ((prediction == 1) & (label == 2)).item()

    def after_test_callback(self, **unused):
        callback_results = {}
        for set_size in self.num_patches:
            callback_results.update({
                f'acc_{set_size}': self.pred_correct[set_size] / self.pred_num[set_size],
                f'hit_rate_{set_size}': self.TP[set_size] / (self.TP[set_size] + self.FN[set_size]),
                f'false_alarm_{set_size}': self.FP[set_size] / (self.FP[set_size] + self.TN[set_size]),
                })
        return callback_results


if __name__ == '__main__':
    # Instantiate the task
    env = ChangeDetection(num_patches=(3, 4))
    display_env_trial(env)
