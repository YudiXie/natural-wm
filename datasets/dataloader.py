import gym
import torch
import torch.utils.data as tud

from tasks.tasktools import ImageTrialEnv

def trial_collate_fn(batch):

    max_len = max([trial[0].shape[0] for trial in batch])
    batch_size = len(batch)
    ob_shape = batch[0][0].shape[1: ]
    gt_shape = batch[0][1].shape[1: ]

    input = torch.zeros(max_len, batch_size, *ob_shape)
    output = torch.zeros(max_len, batch_size, *gt_shape)
    mask = torch.zeros(max_len, batch_size)

    trial_info = []

    for idx, trial in enumerate(batch):
        length = trial[0].shape[0]
        trial[2]['trial_length'] = length
        trial_info.append(trial[2])

        if "output_mask" in trial[2]:
            _mask = trial[2]["output_mask"]
        else:
            _mask = torch.ones((length, ))

        input[: length, idx] = torch.from_numpy(trial[0]) / 255
        output[: length, idx] = torch.from_numpy(trial[1])
        mask[: length, idx] = _mask

    input = input.permute(0, 1, 4, 2, 3)

    return input, output, mask, trial_info


class EnvDataset(tud.Dataset):

    def __init__(self, env: ImageTrialEnv, noise_std=0, noise_mode='per_step', noise_res=None):
        self.env = env
        self.noise_std = noise_std
        self.noise_mode = noise_mode
        self.noise_res = noise_res

    def __len__(self):
        return 100000
        
    def __getitem__(self, idx):
        trial = self.env.new_trial(index=idx)
        self.env.add_input_noise(self.noise_mode, self.noise_std, self.noise_res)
        ob, gt = self.env.ob, self.env.gt
        return ob, gt, trial

class TrialDataset(tud.DataLoader):
    """Make an environment into an iterable dataset for supervised learning.
    Each batch in the dataset i

    Create an iterator that at each call returns
        inputs: numpy array (sequence_length, batch_size, input_units)
        target: numpy array (sequence_length, batch_size, output_units)

    Args:
        env: str for env id or gym.Env objects
        env_kwargs: dict, additional kwargs for environment, if env is str
        batch_size: int, batch size
        seq_len: int, sequence length
    """

    def __init__(self, 
                 env, env_kwargs=dict(), 
                 noise_std=0, noise_mode='per_step', noise_res=None,
                 **dataloader_kwargs):
        
        if not isinstance(env, gym.Env):
            assert isinstance(env, str), 'env must be gym.Env or str'
            if env_kwargs is None:
                env_kwargs = {}
            env = gym.make(env, **env_kwargs)

        env.reset()
        self.env: ImageTrialEnv = env
        dataset = EnvDataset(env, noise_std, noise_mode, noise_res)

        dataloader_kwargs['collate_fn'] = trial_collate_fn

        super().__init__(dataset, **dataloader_kwargs, drop_last=True)
        self.iter = iter(self)

    def __call__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self)
            data = next(self.iter)
        return data