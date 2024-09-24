import logging
import os
import os.path as osp

import torch
import torchvision
import torchvision.transforms as transform
from torch.utils.data import DataLoader, Subset

from configs.config_global import OPENMIND_IMAGENET_PATH, ROOT_DIR
from datasets.continuousreport import (ContinuousReportDataset,
                                       SequentialContinuousReportDataset)
from datasets.luckvogel import LuckVogelClassificationDataset, LuckVogelDataset
from datasets.visual_datasets import (CIFAR10ImgVariationGenerator,
                                      ContrastiveLearningViewGenerator,
                                      get_shuffle_pixel_pipeline_transform,
                                      get_simclr_pipeline_transform,
                                      make_white_noise_dataset)


def get_class_size(dataset_name, inc_targets):
    assert dataset_name != 'Omniglot-Colored', 'Omniglot cannot get class size'
    assert dataset_name != 'WhiteNoise', 'WhiteNoise cannot get class size'
    assert dataset_name != 'CelebA', 'CelebA cannot get class size'

    if dataset_name in ['OneHot',
                        'MNIST',
                        'MNIST-Colored',
                        'CIFAR10',
                        'CIFAR10-SimCLR']:
        class_size = 10
    elif dataset_name == 'CIFAR100':
        class_size = 100
    elif dataset_name == 'ImageNet':
        class_size = 1000
    elif dataset_name == 'SpeechCommands':
        class_size = 35
    else:
        # TODO: need to add code to deal with multiple dataset
        print('Dataset not implemented, or collection of datasets, default to size 10')
        class_size = 10

    if inc_targets is not None:
        new_class_size = len(inc_targets)

    assert new_class_size >= 2, 'class size must => 2'
    return new_class_size, class_size


# TODO: could implement this entire structure as an iterator,
# so that a batch is a list of batches for each dataloader
# to consider reset all iterator at the end of training and testing
# an alternative could be a data iterator that keep iterating though all datasets
# note that the notion of epoch doesn't applies anymore
class DatasetIters(object):
    def __init__(self, config, train_flag, b_size):
        """
        Initialize a list of data loaders and datum sizes
        if only one dataset is specified, return a list containing one data loader
        """
        if type(config.dataset) is list or  type(config.dataset) is tuple:
            assert config.joint_train, "Must use joint training for multi_dataset"
            assert all([type(d_set) is str for d_set in config.dataset]
                       ), 'all dataset must be string'
            dataset_list = config.dataset
        elif type(config.dataset) is str:
            assert not config.joint_train, "Must not use joint training for multi_dataset"
            dataset_list = [config.dataset]
        else:
            raise NotImplementedError('Dataset config not recognized')
        self.num_datasets = len(dataset_list)
        assert len(
            config.mod_w) == self.num_datasets, 'mod_w and dataset len must match'

        self.data_iters = []
        self.iter_lens = []
        self.min_iter_len = None

        self.data_loaders = []
        self.datum_sizes = []
        for d_set in dataset_list:
            data_loader, datum_size = init_single_dataset(d_set, train_flag,
                                                          b_size, config)
            self.data_loaders.append(data_loader)
            self.datum_sizes.append(datum_size)
        self.reset()

    def reset(self):
        # recreate iterator for each of the dataset
        self.data_iters = []
        self.iter_lens = []
        for data_l in self.data_loaders:
            data_iter = iter(data_l)
            self.data_iters.append(data_iter)
            self.iter_lens.append(len(data_iter))
        self.min_iter_len = min(self.iter_lens)


def get_dataset(dataset_name, train_flag, config=None):

    collate_f = None

    if dataset_name == 'MNIST':
        trans = transform.Compose([transform.ToTensor(), ])
        # transform.Normalize((0.1307,), (0.3081,))])

        data_set = torchvision.datasets.MNIST(root=osp.join(ROOT_DIR, 'data'),
                                              train=train_flag, download=True,
                                              transform=trans)
        datum_size = (1, 28, 28)

    elif dataset_name == 'MNIST-flatten':
        trans = transform.Compose([transform.ToTensor(),
                                   transform.Lambda(lambda x: torch.flatten(x))])
        data_set = torchvision.datasets.MNIST(root=osp.join(ROOT_DIR, 'data'),
                                              train=train_flag, download=True,
                                              transform=trans)
        datum_size = (784, )

    elif dataset_name == 'MNIST-Colored':
        trans = transform.Compose([transform.Resize((32, 32)),
                                   transform.Grayscale(num_output_channels=3),
                                   transform.ToTensor()])
        data_set = torchvision.datasets.MNIST(root=osp.join(ROOT_DIR, 'data'),
                                              train=train_flag, download=True,
                                              transform=trans)
        datum_size = (3, 32, 32)

    elif dataset_name == 'Omniglot-Colored':
        # in Omniglot, the color of letter is black, which is inverse to MNIST
        trans = transform.Compose([transform.Resize((32, 32)),
                                   transform.Grayscale(num_output_channels=3),
                                   transform.ToTensor()])
        data_set = torchvision.datasets.Omniglot(root=osp.join(ROOT_DIR, 'data'),
                                                 background=train_flag,
                                                 download=True,
                                                 transform=trans)
        datum_size = (3, 32, 32)

    elif dataset_name == 'CIFAR10':
        # classes = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
        #            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        # trans = transform.Compose([transform.ToTensor(),
        #                            transform.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        trans = transform.ToTensor()
        data_set = torchvision.datasets.CIFAR10(root=osp.join(ROOT_DIR, 'data'),
                                                train=train_flag, download=True,
                                                transform=trans)
        datum_size = (3, 32, 32)

    elif dataset_name == 'CIFAR10-flatten':
        trans = transform.Compose([transform.ToTensor(),
                                   transform.Lambda(lambda x: torch.flatten(x))])
        data_set = torchvision.datasets.CIFAR10(root=osp.join(ROOT_DIR, 'data'),
                                                train=train_flag, download=True,
                                                transform=trans)
        datum_size = (3072, )

    elif dataset_name == 'CIFAR10-SimCLR':
        assert train_flag is True, 'SimCLR must use train data'
        trans = ContrastiveLearningViewGenerator(
            get_simclr_pipeline_transform(32))
        data_set = torchvision.datasets.CIFAR10(root=osp.join(ROOT_DIR, 'data'),
                                                train=True, download=True,
                                                transform=trans)
        datum_size = (3, 32, 32)

    elif dataset_name == 'LuckVogel-SimCLR':
        assert train_flag is True, 'SimCLR must use train data'

        trans = ContrastiveLearningViewGenerator(
            transform.Compose([
                transform.ToPILImage(), 
                get_simclr_pipeline_transform(32, s=0) # Avoid color jittoring here
            ]) 
        )
        data_set = LuckVogelClassificationDataset(config, transform=trans)
        datum_size = tuple(data_set.img_size)

    elif dataset_name == 'CIFAR10-Shuffle':
        trans = CIFAR10ImgVariationGenerator(get_shuffle_pixel_pipeline_transform())
        data_set = torchvision.datasets.CIFAR10(root=osp.join(ROOT_DIR, 'data'),
                                                train=True, download=True,
                                                transform=trans)
        datum_size = (3, 32, 32)

    elif dataset_name == 'CIFAR100':
        trans = transform.ToTensor()
        data_set = torchvision.datasets.CIFAR100(root=osp.join(ROOT_DIR, 'data'),
                                                 train=train_flag, download=True,
                                                 transform=trans)
        datum_size = (3, 32, 32)

    elif dataset_name == 'CelebA':
        if train_flag:
            spl = 'train'
        else:
            spl = 'valid'
        trans = transform.Compose([transform.Resize((32, 32)),
                                   transform.ToTensor()])
        data_set = torchvision.datasets.CelebA(root=osp.join(ROOT_DIR, 'data'),
                                               split=spl, download=True,
                                               transform=trans)
        datum_size = (3, 32, 32)

    elif dataset_name == 'ImageNet':

        if train_flag:
            spl = 'train'
        else:
            spl = 'val'
        trans = transform.Compose([transform.Resize(256),
                                   transform.CenterCrop(224),
                                   transform.ToTensor(),
                                   transform.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])])
        data_set = torchvision.datasets.ImageNet(root=OPENMIND_IMAGENET_PATH,
                                                 split=spl, transform=trans)
        datum_size = (3, 224, 224)

    elif dataset_name == 'WhiteNoise':
        # WhiteNoise is a dataset that contain 10000 white noise images
        # Cropped to (3, 32, 32)
        # TODO: Normalized to have the same statistics as CIFAR10
        trans = transform.Compose([transform.ToTensor(), ])
        dataset_path = osp.join(ROOT_DIR, 'data', 'WhiteNoise')
        # If dataset doesn't exist, then create it.
        if not os.path.isdir(dataset_path):
            make_white_noise_dataset(data_path=dataset_path)

        data_set = torchvision.datasets.ImageFolder(root=dataset_path,
                                                    transform=trans)
        assert len(data_set) >= 10000, 'Dataset should has at least 10000 images'

        datum_size = (3, 32, 32)

    elif dataset_name == 'WhiteNoiseSingleChannel':
        # WhiteNoise is a dataset that contain 10000 white noise images
        # Cropped to (1, 28, 28)
        # TODO: Normalized to have the same statistics as MNIST
        trans = transform.Compose([transform.ToTensor(),
                                   transform.Grayscale(num_output_channels=1)])
        dataset_path = osp.join(ROOT_DIR, 'data', 'WhiteNoiseSingleChannel')
        # If dataset doesn't exist, then create it.
        if not os.path.isdir(dataset_path):
            make_white_noise_dataset(
                data_path=dataset_path, img_size=(3, 28, 28))

        data_set = torchvision.datasets.ImageFolder(root=dataset_path,
                                                    transform=trans)
        assert len(data_set) >= 10000, 'Dataset should has at least 10000 images'

        datum_size = (1, 28, 28)

    elif dataset_name == 'LuckVogel':
        data_set = LuckVogelDataset(config)
        datum_size = tuple(data_set.img_size)

    elif dataset_name == 'LuckVogelClassification':
        data_set = LuckVogelClassificationDataset(config)
        datum_size = tuple(data_set.img_size)

    elif dataset_name == 'ContinuousReportDataset':
        data_set = ContinuousReportDataset(config)
        datum_size = tuple(data_set.img_size)

    elif dataset_name == 'SequentialContinuousReportDataset':
        data_set = SequentialContinuousReportDataset(config)
        datum_size = tuple(data_set.img_size)
    else:
        raise NotImplementedError('Dataset not implemented')

    return data_set, collate_f, datum_size

# TODO: should separate datum size and dataloader initialization
# could use a list of datum size to initialize a model and only train or test with one dataset
def init_single_dataset(dataset_name, train_flag, b_size, config=None):
    
    data_set, collate_f, datum_size = get_dataset(dataset_name, train_flag, config)

    if config is not None:
        num_wks = config.num_workers
    else:
        num_wks = 0

    data_loader = DataLoader(data_set, batch_size=b_size, shuffle=train_flag,
                             num_workers=num_wks, drop_last=True, collate_fn=collate_f)

    # print("datum size:", datum_size)
    return data_loader, datum_size
