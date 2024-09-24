import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from utils.model_utils import get_cnn
from tasks.changedetection import ChangeDetection
from configs.config_global import FIG_DIR

matplotlib.rcParams['figure.facecolor'] = 'w'
plt.rcParams.update({'font.size': 16})

def adjust_figure():
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout(pad=0.5)

if __name__ == '__main__':
    net = get_cnn('pretResNet', (3, 32, 32), 0, False, 'instancenorm',
                  64, 'CIFAR10', True)
    embed_num = 100
    embed_diff_norm_list = []
    num_patches = np.arange(1, 13)
    for num_patch in num_patches:
        env = ChangeDetection(num_patches=(num_patch,), use_fixed_colors=True)

        embed_sample = np.zeros((embed_num, 64))
        embed_test = np.zeros((embed_num, 64))

        i_trial = 0
        while i_trial < embed_num:
            trial = env.new_trial()
            # if non-match
            if trial['ground_truth'] == 2:
                # sample image
                image1 = torch.from_numpy(env.ob[0]) / 255
                image1 = image1.permute(2, 0, 1).unsqueeze(0)

                # test image
                image2 = torch.from_numpy(env.ob[-1]) / 255
                image2 = image2.permute(2, 0, 1).unsqueeze(0)

                embed1 = net(image1).squeeze().numpy()
                embed2 = net(image2).squeeze().numpy()

                embed_sample[i_trial] = embed1
                embed_test[i_trial] = embed2
                i_trial += 1

        embed_diff = embed_sample - embed_test
        embed_diff_norm = np.linalg.norm(embed_diff, axis=1)
        embed_diff_norm_list.append(embed_diff_norm)

    plt.boxplot(embed_diff_norm_list, notch=True)
    plt.ylabel("sample - test embedding L2 norm")
    plt.xlabel('Set size')
    adjust_figure()
    plt.savefig(os.path.join(FIG_DIR, 'embed_difference.pdf'), transparent=True)

