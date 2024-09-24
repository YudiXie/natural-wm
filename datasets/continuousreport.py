import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np
from configs.configs import ContinuousReportConfig

from datasets.luckvogel import generate_pos, lg_gray_color
from tasks.tasktools import angle2color, random

# black color
black_color = torch.tensor([0, 0, 0]) / 255

class ContinuousReportDataset(Dataset):
    """ContinuousReport dataset"""

    def __init__(self, config: ContinuousReportConfig):
        """
        Args:
            num_patch (int): number of patches in a single image
        """

        self.img_size = config.img_size
        self.num_patch = config.num_patches
        self.patch_size = config.patch_size

        if isinstance(self.num_patch, int):
            self.num_patch = (self.num_patch, )
        assert max(self.num_patch) <= 12

        self.data_set_len = 5000
        self.all_possible_probes = config.all_possible_probes

        if self.all_possible_probes:
            assert len(self.num_patch) == 1, "In the test mode, set size must be fixed"

    def __len__(self):
        return self.data_set_len

    def get_cue(self, positions, angles, cue_idx):
        # cue image
        cue_image = lg_gray_color[0] * torch.ones(self.img_size)

        # randomly select one patch to cue
        cue_pos = positions[cue_idx]
        patch = black_color[:, None, None].repeat(1, self.patch_size[0], self.patch_size[1])
        cue_image[:, cue_pos[0]: cue_pos[0] + self.patch_size[0],
                     cue_pos[1]: cue_pos[1] + self.patch_size[1]] = patch

        angle = torch.tensor(angles[cue_idx])
        cue_image = torch.Tensor(cue_image)

        return angle, cue_image

    def __getitem__(self, idx, positions=None, angles=None):

        if positions is None:
            num_patch = self.num_patch[idx % len(self.num_patch)]
            positions = generate_pos(self.img_size, self.patch_size, num_patch)
        else:
            num_patch = len(positions)

        # sample image
        image = lg_gray_color[0] * torch.ones(self.img_size)

        if angles is None:
            angles = []
            for pos in positions:
                # sample a random angle from the color wheel
                angle = 2 * np.pi * random.rand() - np.pi
                angles.append(angle)

        for pos, angle in zip(positions, angles):
            rgb_color = torch.tensor(angle2color(angle))
            patch = rgb_color[:, None, None].repeat(1, self.patch_size[0], self.patch_size[1])
            image[:, pos[0]: pos[0] + self.patch_size[0],
                     pos[1]: pos[1] + self.patch_size[1]] = patch
        
        image = torch.Tensor(image)

        if not self.all_possible_probes:
            cue_idx = 0
            angle, cue_image = self.get_cue(positions, angles, cue_idx)

            d_len = max(self.num_patch) - len(positions)
            positions += d_len * [(-1, -1), ]
            angles += d_len * [-10, ]

            return image, cue_image, angle, num_patch, cue_idx, (np.array(positions), np.array(angles))

        else:
            n = self.num_patch[0]
            angle_list = []
            cue_image_list = []

            for cue_idx in range(n):
                angle, cue_image = self.get_cue(positions, angles, cue_idx)
                angle_list.append(angle)
                cue_image_list.append(cue_image)

            angle_list = torch.stack(angle_list)
            cue_image_list = torch.stack(cue_image_list)
            image_list = image.repeat(n, 1, 1, 1)

            return image_list, cue_image_list, angle_list, torch.full((n, ), n)

class SequentialContinuousReportDataset(ContinuousReportDataset):

    def __init__(self, config):
        super().__init__(config)

        self.sample_step = config.sample_step
        self.interval_step = config.interval_step
        self.delay_step = config.delay_step

        if isinstance(self.delay_step, int):
            self.delay_step = (self.delay_step, )

        self.test_step = config.test_step
        self.max_seq_len = (self.sample_step + self.interval_step) * max(self.num_patch) + max(self.delay_step) + self.test_step

    def get_seq(self, positions, angles, cue_idx, delay_step, num_patch, images):

        angle, cue_image = self.get_cue(positions, angles, cue_idx)

        fixation_image = lg_gray_color[0] * torch.ones(self.img_size)

        # create a mask that corresponds to the testing stage
        target_mask = torch.zeros((self.max_seq_len, ), dtype=torch.bool)
        test_start = num_patch * (self.sample_step + self.interval_step) + delay_step
        target_mask[test_start + self.test_step - 1] = 1

        # construct the sequence of stimuli
        seq = []
        for stimulus in images:
            seq += [stimulus, ] * self.sample_step + [fixation_image, ] * self.interval_step
        seq += [fixation_image, ] * delay_step + [cue_image, ] * self.test_step
        seq += [fixation_image, ] * (self.max_seq_len - len(seq))

        return torch.stack(seq), target_mask, angle

    def __getitem__(self, idx):
        num_patch = self.num_patch[idx % len(self.num_patch)]
        positions = generate_pos(self.img_size, self.patch_size, num_patch)

        # sample image
        images = []
        angles = []

        for pos in positions:
            # sample a random angle from the color wheel
            angle = 2 * np.pi * random.rand() - np.pi
            angles.append(angle)
            rgb_color = torch.tensor(angle2color(angle))
            patch = rgb_color[:, None, None].repeat(1, self.patch_size[0], self.patch_size[1])
            image = lg_gray_color[0] * torch.ones(self.img_size)
            image[:, pos[0]: pos[0] + self.patch_size[0],
                     pos[1]: pos[1] + self.patch_size[1]] = patch

            images.append(image)

        # randomly choose the delay step
        delay_step = random.choice(self.delay_step)

        if not self.all_possible_probes:
            cue_idx = random.randint(len(positions))

            seq, target_mask, angle = self.get_seq(
                positions, angles, cue_idx, 
                delay_step, num_patch, images
            )
            
            # import analysis.plots as plots
            # plots.plot_gif(seq, 'continuous_report_samples', f'seq_sample_{idx}', interval=1000)
            # if idx > 5:
            #     exit(0)

            d_len = max(self.num_patch) - len(positions)
            positions += d_len * [(-1, -1), ]
            angles += d_len * [-1, ]

            return seq, target_mask, angle, num_patch, cue_idx, (np.array(positions), np.array(angles))
            
        else:

            n = self.num_patch[0]
            angle_list = []
            seq_list = []
            target_mask_list = []

            for cue_idx in range(n):
                seq, target_mask, angle = self.get_seq(
                    positions, angles, cue_idx, 
                    delay_step, num_patch, images
                )
                angle_list.append(angle)
                seq_list.append(seq)
                target_mask_list.append(target_mask)

            angle_list = torch.stack(angle_list)
            seq_list = torch.stack(seq_list)
            target_mask_list = torch.stack(target_mask_list)

            return seq_list, target_mask_list, angle_list, torch.full((n, ), n), torch.from_numpy(np.arange(n))
