import os.path as osp
from collections import OrderedDict
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.module import Module
import torchvision.transforms.functional as F
from models.deepdive.model_options import *

from configs.config_global import DEVICE, MAP_LOC, ROOT_DIR
from configs.configs import BaseConfig
from models.attention_modules import *
from models.cognitive_model import VanillaRNNCell
from models.sensory_model import M5, LeNet, MultiscaleResNet, ResNet
from tasks.tasktools import random

import torchvision.models as tvmodels


def remove_backbone(old_key):
    return old_key[9:]

def get_rnn_para(old_state_dict):

    new_state_dict = OrderedDict()
    for key, value in old_state_dict.items():
        if 'rnn' in key:
            new_key = key[4:]
            new_state_dict[new_key] = value

    return new_state_dict


def get_out_layer_para(old_state_dict):
    new_state_dict = OrderedDict()
    for key, value in old_state_dict.items():
        if 'out_layer' in key:
            new_key = key[10:]
            new_state_dict[new_key] = value

    return new_state_dict


def get_model_state_dict(config, net_number):
    if net_number is not None:
        model_path = osp.join(config.save_path, 
                              'net_' + str(net_number) + '.pth')
    else:
        model_path = osp.join(config.save_path, 'net_best.pth')
    state_dict = torch.load(model_path, map_location=MAP_LOC)
    return state_dict, model_path


def get_pret_cnn_state_dict(config):
    seed = config.seed if config.different_pretrain_seeds else 0
    cnn_path = get_pret_cnn_path(config.cnn_archi,
                                 config.cnn_pret,
                                 cnn_norm=config.cnn_norm,
                                 cnn_width=config.cnn_width,
                                 seed=seed
                                )
    state_dict = torch.load(cnn_path, map_location=MAP_LOC)
    # only load the weights up to the second-to-last layer
    state_dict.pop('out_layer.weight')
    state_dict.pop('out_layer.bias')
    return state_dict, cnn_path


def get_pret_audicnn_state_dict(model_type='M5', seed=0):
    cnn_path = osp.join(ROOT_DIR, 'experiments',
                        'audio_classification_pretrain',
                        f'model_model_type{model_type}_s{seed}',
                        'net_best.pth')
    state_dict = torch.load(cnn_path, map_location=MAP_LOC)
    # only load the weights up to the second-to-last layer
    state_dict.pop('fc1.weight')
    state_dict.pop('fc1.bias')

    return state_dict, cnn_path


def rename_state_dict_keys(old_state_dict, key_transformation):
    """
    old_state_dict: Path to the saved state dict.
    key_transformation: Function that accepts the old key names of the state
                        dict as the only argument and returns the new key name.
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    new_state_dict = OrderedDict()
    for key, value in old_state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    return new_state_dict


def get_pret_cnn_path(cnn_archi: str,
                      cnn_pret: str,
                      cnn_norm: str = 'batchnorm', 
                      cnn_width: int = 64, 
                      seed: int = 0, 
                      config = None) -> str:
    """
    get the path of pretrained cnn weights
    Args:
        cnn_archi: CNN architecture
        cnn_pret: CNN pretraining paradigm
        cnn_norm: CNN normalization
        cnn_width: CNN width
        seed: random seed
        config: config object
    """
    if cnn_archi == 'LeNet':
        raise NotImplementedError('LeNet pretrained model not implemented')
    elif cnn_archi in ['ResNet', 'MultiscaleResNet']:
        if cnn_pret == 'Classification_MNIST':
            ret_path = osp.join(ROOT_DIR, 'experiments', 
                                'classification_pretrain_mnist',
                                f'model_cnn_norm{cnn_norm}_cnn_width{cnn_width}_s{seed}', 
                                'net_best.pth')
        elif cnn_pret == 'Classification_CIFAR10':
            ret_path = osp.join(ROOT_DIR, 'experiments', 
                                'classification_pretrain',
                                f'model_cnn_norm{cnn_norm}_cnn_width{cnn_width}_s{seed}', 
                                'net_best.pth')
        elif cnn_pret == 'Classification_LuckVogel':
            ret_path = osp.join(ROOT_DIR, 'experiments', 
                                'luckvogel_classification_pretrain',
                                f'model_cnn_norm{cnn_norm}_cnn_width{cnn_width}_s{seed}', 
                                'net_best.pth')
        elif cnn_pret == 'Contrastive_CIFAR10':
            ret_path = osp.join(ROOT_DIR, 'experiments', 
                    'contrastive_pretrain',
                    f'model_cnn_norm{cnn_norm}_cnn_width{cnn_width}_s{seed}', 
                    'net_best.pth')
        elif cnn_pret == 'Contrastive_LuckVogel':
            ret_path = osp.join(ROOT_DIR, 'experiments', 
                                'luckvogel_contrastive_pretrain',
                                f'model_cnn_norm{cnn_norm}_cnn_width{cnn_width}_' +
                                f'max_pretrain_set_size{config.max_pretrain_set_size}_s{seed}', 
                                'net_best.pth')
        else:
            raise NotImplementedError('No pretrained model')
    else:
        raise NotImplementedError('Model architecture not implemented')

    return ret_path

# import torchvision.transforms._presets

def imagenet_transform(x):
    """
    x: a batch of images, values should be in [0, 1]
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = 224
    # See 
    x = F.resize(x, size)
    if not isinstance(x, torch.Tensor):
        x = F.to_tensor(x)
    x = F.normalize(x, mean=mean, std=std)
    return x

def taskonomy_transform(x):
    size = 256
    x = F.resize(x, size)
    x = x * 2 - 1
    return x

def get_cnn(config: BaseConfig, image_size=None, skip_last=True):
    """
    get the cnn model
    Args:
        config: config object
        image_size: image size
        skip_last: output of CNN is the last layer if True, 
            else output the second-to-last-layer
    """
    cnn_archi = config.cnn_archi
    cnn_pret = config.cnn_pret

    seed = config.seed if config.different_pretrain_seeds else 0
    preprocess = None

    if cnn_archi == 'Identity':
        # When embedding is precomputed as input, use identity as 'encoder'
        cnn = nn.Identity()

        if cnn_pret != 'none':
            raise NotImplementedError('Identity can not be pretrained')

    elif cnn_archi == 'LeNet':
        cnn = LeNet(image_size, config.model_class_size)
        if cnn_pret != 'none':
            raise NotImplementedError('LeNet pretrained model not implemented')

        if skip_last:
            # output the second-to-last-layer
            cnn.out_layer = nn.Identity()

    elif cnn_archi in ['ResNet', 'MultiscaleResNet']:
        model_class = ResNet if cnn_archi == 'ResNet' else MultiscaleResNet
        cnn = model_class(
            config.resblock_config, 
            image_size, 
            norm_layer=config.cnn_norm,
            width=config.cnn_width,
            spatial_average=config.spatial_average,
            num_classes=config.model_class_size
        )

        if skip_last:
            cnn.out_layer = nn.Identity()

        if cnn_pret != 'none':
            path = get_pret_cnn_path(
                cnn_archi, 
                cnn_pret,
                config.cnn_norm, 
                config.cnn_width,
                seed=seed,
                config=config
            )

            state_dict = torch.load(path, map_location=DEVICE)

            # only load the weights up to the second-to-last layer
            if cnn_pret in ['Contrastive_CIFAR10', 'Contrastive_LuckVogel']:
                state_dict = rename_state_dict_keys(state_dict, 
                                                    remove_backbone)
                state_dict.pop('out_layer.0.weight')
                state_dict.pop('out_layer.0.bias')
                state_dict.pop('out_layer.2.weight')
                state_dict.pop('out_layer.2.bias')
            else:
                state_dict.pop('out_layer.weight')
                state_dict.pop('out_layer.bias')

            ret = cnn.load_state_dict(state_dict, strict=False)
            print(f'Loaded CNN weights from {path}')
            print(f'Missing keys: {ret[0]}, Unexpected Keys: {ret[1]}')
    
    elif cnn_archi in ['ResNet-18', 'ResNet-50']:
        if cnn_pret in ['Classification_ImageNet', 'none']:
            if cnn_archi == 'ResNet-18':
                model_handle = tvmodels.resnet18
                model_weights = tvmodels.ResNet18_Weights.IMAGENET1K_V1
            elif cnn_archi == 'ResNet-50':
                model_handle = tvmodels.resnet50
                model_weights = tvmodels.ResNet50_Weights.IMAGENET1K_V2
            else:
                raise NotImplementedError(f'{cnn_archi} not implemented')

            cnn = model_handle(weights=None)
            if cnn_pret != 'none':
                assert cnn_pret == 'Classification_ImageNet'
                cnn = model_handle(weights=model_weights)
                preprocess = imagenet_transform

            if skip_last:
                cnn.fc = nn.Identity()
        elif cnn_pret == 'R3M':
            from r3m import load_r3m
            cnn = load_r3m(f"resnet{cnn_archi[-2:]}") # resnet18, resnet50
            preprocess = lambda x: x * 255.0
        elif cnn_pret[-9: ] == 'taskonomy':
            assert cnn_archi == 'ResNet-50', 'Only ResNet-50 is supported for taskonomy'
            options = get_model_options(model_source='taskonomy')
            cnn = eval(options[cnn_pret]['call'])
            preprocess = taskonomy_transform
        else:
            raise NotImplementedError('No pretrained model')

    elif cnn_archi == 'AlexNet':
        cnn = tvmodels.alexnet(weights=None)
        if cnn_pret != 'none':
            assert cnn_pret == 'Classification_ImageNet'
            cnn = tvmodels.alexnet(weights=tvmodels.AlexNet_Weights.IMAGENET1K_V1)
            preprocess = imagenet_transform

        if skip_last:
            cnn.classifier[-1] = nn.Identity()

    elif cnn_archi == 'ViT-B':
        if cnn_pret == 'none':
            cnn = tvmodels.vit_b_32(weights=None)
        elif cnn_pret == 'Classification_ImageNet':
            cnn = tvmodels.vit_b_32(weights=tvmodels.ViT_B_32_Weights.IMAGENET1K_V1)
            preprocess = imagenet_transform
        elif cnn_pret == 'VC-1':
            from vc_models.models.vit import model_utils
            cnn, embd_size, preprocess, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
        else:
            raise NotImplementedError('No pretrained model')

        if skip_last and cnn_pret in ['none', 'Classification_ImageNet']:
            cnn.heads[-1] = nn.Identity()

    elif cnn_archi == 'M5':
        # Auditory Encoders
        cnn = M5()

        if skip_last:
            cnn.fc1 = nn.Identity()

        if cnn_pret != 'none':
            state_dict, _ = get_pret_audicnn_state_dict(seed=seed)
            cnn.load_state_dict(state_dict, strict=True)
        
    else:
        raise NotImplementedError('CNN not implemented')
    
    
    class Wrapper(nn.Module):
        def __init__(self, cnn):
            super().__init__()
            self.cnn = cnn

        def forward(self, x):
            if preprocess is not None:
                x = preprocess(x)
            x = self.cnn(x)
            if config.embedding_size is not None:
                x = x[:, : config.embedding_size]
            return x
        
    cnn = Wrapper(cnn)
    
    cnn.requires_grad_(not config.freeze_cnn)
    cnn.train(not config.freeze_cnn)

    return cnn


def get_rnn(rnn_type, rnn_in_size, hidden_size, alpha=0.1, layernorm=False, add_noise=0, mul_noise=0):
    
    if rnn_type == 'VanillaRNN':
        rnn = VanillaRNNCell(
            rnn_in_size, hidden_size, 
            add_rnn_noise=add_noise, mul_rnn_noise=mul_noise
        )
    elif rnn_type == 'CTRNN':
        rnn = VanillaRNNCell(
            rnn_in_size, hidden_size, 
            ct=True, alpha=alpha, layernorm=layernorm, 
            add_rnn_noise=add_noise, mul_rnn_noise=mul_noise
        )
    elif rnn_type == 'LSTM':
        rnn = nn.LSTMCell(rnn_in_size, hidden_size)
    elif rnn_type == 'GRU':
        rnn = nn.GRUCell(rnn_in_size, hidden_size)
    else:
        raise NotImplementedError('RNN not implemented')

    return rnn

def get_att(att_type, hidden_size, feature_shape):

    if att_type == 'film':
        att = FiLM_att(hidden_size, feature_shape[0])
    elif att_type == 'spatial':
        att = Spatial_att(hidden_size, *feature_shape)
    elif att_type == 'none':
        att = Identity_att()
    elif att_type == 'elementwise':
        att = Elementwise_att(hidden_size, *feature_shape)
    elif att_type == 'cbam':
        att = CBAM_att(hidden_size, *feature_shape)
    elif att_type == 'feature':
        att = Feature_att(hidden_size, feature_shape[0])
    elif att_type == 'cbam_spatial':
        att = CBAM_Spatial_att(hidden_size, *feature_shape)
    else:
        raise NotImplementedError('Attention not implemented')

    return att


def get_pret_crnn_path(dataset_n):
    if dataset_n == 'MNIST':
        ret_path = osp.join(ROOT_DIR, 'experiments', 'mem_pres_pretrain2',
                            'model_model_typeConvRNNBL_t_array_5_5__datasetMNIST', 'net_best.pth')
    elif dataset_n == 'CIFAR10':
        ret_path = osp.join(ROOT_DIR, 'experiments', 'mem_pres_pretrain2',
                            'model_model_typeConvRNNBL_t_array_5_5__datasetCIFAR10', 'net_best.pth')
    else:
        raise NotImplementedError('pretrained model not implemented')

    return ret_path
