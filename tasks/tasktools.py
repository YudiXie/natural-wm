import colorsys
import math
from typing import Any, List

import neurogym as ngym
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from skimage.color import lab2rgb
import gym

from configs.configs import BaseConfig
from utils.logger import Logger

class random:

    @staticmethod
    def randint(low, high=None):
        """
        generate a random integer from [0, low) or [low, high) using the torch package
        """
        if high is None:
            high = low
            low = 0
        return torch.randint(low=low, high=high, size=()).item()

    @staticmethod
    def rand():
        return torch.rand(size=()).item()
    
    @staticmethod
    def randn():
        return torch.randn(size=()).item()

    @staticmethod
    def choice(l):
        return l[random.randint(len(l))]
    
    @staticmethod
    def rand_like(x):
        if isinstance(x, torch.Tensor):
            return torch.rand_like(x)
        elif isinstance(x, np.ndarray):
            return torch.rand(*x.shape).numpy()
        else:
            raise ValueError('x must be a torch.Tensor or np.ndarray')
    
    @staticmethod
    def randn_like(x):
        if isinstance(x, torch.Tensor):
            return torch.randn_like(x)
        elif isinstance(x, np.ndarray):
            return torch.randn(*x.shape).numpy()
        else:
            raise ValueError('x must be a torch.Tensor or np.ndarray')

def get_noise(tensor: torch.Tensor, noise_std, noise_res=None):
    if isinstance(noise_std, (list, tuple)):
        noise_std = random.choice(noise_std)
    if noise_res is not None:
        noise = torch.randn(*tensor.shape[: -2], noise_res, noise_res, device=tensor.device)
        if noise_res != tensor.shape[-1]:
            noise = TF.resize(noise, size=tensor.shape[-2:])
    else:
        noise = torch.randn_like(tensor)
    return noise * noise_std

def add_noise(tensor: torch.Tensor, noise_std, noise_res=None):
    return tensor + get_noise(tensor, noise_std, noise_res)

class ImageTrialEnv(ngym.TrialEnv):
    """Superclass for environments with image stimuli."""

    def add_ob(self, value, period=None, where=None):
        if where is not None:
            print('Warning: Adding where to values other than None'
                  'has no effect when adding image stimuli')

        super().add_ob(value, period, where)

    def set_ob(self, value, period=None, where=None):
        if where is not None:
            print('Warning: Setting where to values other than None'
                  'has no effect when adding image stimuli')

        super().set_ob(value, period, where)

    def set_mask(self, trial, value, period=None):

        if "output_mask" not in trial:
            tmax_ind = int(self._tmax / self.dt)
            trial["output_mask"] = torch.zeros((tmax_ind, ), )

        trial["output_mask"][self.start_ind[period]: self.end_ind[period]] = value

    def before_test_callback(self, model: nn.Module, save_path: str, batch_num: int):
        """
        An optional function called before the test stage, possibly to initialize infomation
        """
        pass

    def add_input_noise(self, mode: str='per_step', noise_std: float=0, noise_res: int = None):
        """
        Add noise to self.ob, note that after adding noise, pixel values might not be in [0, 255] and are no longer integers
        :param mode: 'per_step', 'per_period' or 'per_trial', whether to add independent noise to each time step, each period or each trial
        :param noise_std: the standard deviation of the noise devided by 255.0
        :param noise_res: the resolution of the noise, if None, the noise has the same resolution as the input
        """
        assert noise_res == None
        self.ob = self.ob.astype(np.float32)
        noise_std *= 255.0
        if mode == 'per_step':
            self.ob += random.randn_like(self.ob) * noise_std
        elif mode == 'per_period':
            for period in self.start_ind.keys():
                self.ob[self.start_ind[period]: self.end_ind[period]] += random.randn_like(self.ob[0]) * noise_std
        elif mode == 'per_trial':
            self.ob += random.randn_like(self.ob[0]) * noise_std
        else:
            raise ValueError('mode must be one of "per_step", "per_period" and "per_trial"')

    def test_step(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, 
                    loss_mask: torch.Tensor, trial_info: List[dict], 
                    hiddens: List[torch.Tensor], outputs: List[torch.Tensor]):
        """
        :param inputs: Tensor of shape [trial_length, batch_size, *input_shape]
        :param loss_mask: Tensor of shape [trial_length, batch_size]
        :param hiddens: The hidden state for each time step
        :param outputs: The hidden state 
        """
        pass

    def after_test_callback(self, model: nn.Module, save_path: str, is_best: bool, batch_num: int):
        """
        An optional function called after the test stage
        :param is_best: whether the test result is the best one
        :param batch_num: number of batches used for training
        return: a dictionary of information to be logged
        """
        pass

    def after_training_callback(self, config: BaseConfig, model: nn.Module):
        """
        An optional function called after training is done
        :param model: the best model
        """
        pass

    def criterion(self, output, target):
        raise NotImplementedError('criterion must be specified')
    
class ImageTrialEnv_Wrapper(gym.Wrapper):

    def __init__(self, env: ImageTrialEnv):
        self.env = env

    def __getattr__(self, __name: str) -> Any:
        return self.env.__getattribute__(__name)

def angle2color(angle, colorspace='CIELAB'):
    """
    Convert an angle on a color ring to a RGB color.
    args:
        angle, is within [-pi, pi), if in range
        colorspace, either 'HSV' or 'CIELAB'
    return:
        ndarray, a color rgb code, each element in range [0, 1]
    """
    if colorspace == 'HSV':
        color_code_h = (angle + np.pi) / (2.0 * np.pi)
        color = np.array(colorsys.hsv_to_rgb(color_code_h, 1, 1.0))
    elif colorspace == 'CIELAB':
        # CIELAB color ring centered at L = 54, a = 21.5, and b = 11.5, radius = 49
        # implemented as in Schurgin, M.W., Wixted, J.T. and Brady, T.F., 2020.
        # Psychophysical scaling reveals a unified theory of visual
        # memory strength. Nature human behaviour, 4(11), pp.1156-1172.
        l, radius = 54, 49
        a = 21.5 + radius * np.cos(angle)
        b = 11.5 + radius * np.sin(angle)
        color = lab2rgb([l, a, b])
    else:
        raise NotImplementedError(f'Color space: {colorspace} is not implemented')
    return color


def angular_distance(angle1, angle2):
    """
    Compute the angular distance between two angles.
    args:
        angle1: an angle, in range [-pi, pi)
        angle2: an angle, in range [-pi, pi)
    return:
        the angular distance, in range [0, pi]
    """
    return np.abs((angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi)


def sample_color_angles(num_of_angles, rng=None, minimal_pairwise_distance=np.pi / 12, possible_colors=360):
    """
    Sample a list of color angles that are dissimilar (at least 15 deg apart) to each other.
    args:
        num_of_angles: number of angles to sample
        rng: a numpy.random.RandomState object
    :return:
        a list of color angles in range [-pi, pi)
    """
    if rng is None:
        rng = random
    assert num_of_angles <= 12, "Number of angles should be less than or equal to 12"

    angles = []
    for i in range(num_of_angles):
        while True:
            angle = 2 * np.pi * (rng.randint(possible_colors) / possible_colors) - np.pi
            if all([angular_distance(angle, prev_angle) > minimal_pairwise_distance for prev_angle in angles]):
                break
        angles.append(angle)
    return angles

def get_changed_angle(angle, delta):
    """
    Get a new angle whose angular distance to the original angle is delta
    args:
        angle: an angle, in range [-pi, pi)
    """
    if random.randint(2):
        angle = angle + delta
    else:
        angle = angle - delta
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

def get_opposite_angle(angle):
    """
    Get the opposite angle of a given angle.
    args:
        angle: an angle, in range [-pi, pi)
    return:
        the opposite angle, in range [-pi, pi)
    """
    return get_changed_angle(angle, np.pi)


def sample_target_foil_angles(rng=None, mode='random'):
    """
    Sample a target and a foil angle that are maximally dissimilar or randomly sampled.
    args:
        rng: a numpy.random.RandomState object
        mode: 'random' or 'dissimilar',
            random: randomly sample target and foil angles, but at least 30 deg apart
            dissimilar: sample target and foil angles that are maximally dissimilar (180 deg apart)
            similar: sample target and foil angles that similar (less than 30 deg apart)
    return:
        target_angle: a target angle, in range [-pi, pi)
        foil_angle: a foil angle, in range [-pi, pi)
    """
    if rng is None:
        rng = np.random.RandomState()

    target_angle = 2 * np.pi * rng.rand() - np.pi
    if mode == 'dissimilar':
        foil_angle = get_opposite_angle(target_angle)
    elif mode == 'random':
        while True:
            foil_angle = 2 * np.pi * rng.rand() - np.pi
            if angular_distance(target_angle, foil_angle) > np.pi / 6:
                break
    elif mode == 'similar':
        while True:
            foil_angle = 2 * np.pi * rng.rand() - np.pi
            if angular_distance(target_angle, foil_angle) < np.pi / 6:
                break
    else:
        raise ValueError("Mode should be either 'random' or 'maximal'")
    return target_angle, foil_angle


def show_trial(observation):
    """
    To visualize a trial of an image environment.
    Args:
        observation: numpy array of shape (T, H, W, C)
    """
    import matplotlib.pyplot as plt

    # plot the image stimuli in a trial in 3 rows
    n_row = 3
    n_col = math.ceil(observation.shape[0] / n_row)
    fig, axs = plt.subplots(n_row, n_col)

    # plot the image stimuli
    for i in range(observation.shape[0]):
        i_row = i // n_col
        i_col = i % n_col
        axs[i_row, i_col].imshow(observation[i])

    # get rid of unnecessary axes
    for i in range(observation.shape[0], n_row * n_col):
        i_row = i // n_col
        i_col = i % n_col
        axs[i_row, i_col].axis('off')
    plt.show()


def display_env_trial(env):
    """To visualize a trial of an image environment."""
    assert isinstance(env, ImageTrialEnv), 'env must be an ImageTrialEnv'
    trial = env.new_trial()
    print('Trial info', trial)
    print('Trial observation shape', env.ob.shape)
    print('Trial ground truth action shape', env.gt.shape)
    print('Trial ground truth action', env.gt)

    show_trial(env.ob)

    env.reset()
    ob, reward, done, info = env.step(env.action_space.sample())
    print('Single time step observation shape', ob.shape)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    theta = sample_color_angles(12)
    r = np.ones((len(theta), ))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(theta, r)
    ax.set_rmax(2)
    ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()
