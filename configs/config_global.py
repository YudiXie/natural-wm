import os.path as osp
import logging
import torch

NP_SEED = 1234
TCH_SEED = 2147483647

# config for ConvRNNs
ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
FIG_DIR = osp.join(ROOT_DIR, 'figures')
DATA_DIR = osp.join(ROOT_DIR, 'data')

# device to run algorithm on
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MAP_LOC = "cuda:0" if USE_CUDA else torch.device('cpu')
LOG_LEVEL = logging.INFO
ANA_LOG_LEVEL = logging.WARNING

OPENMIND_IMAGENET_PATH = '/om2/user/yu_xie/data/imagenet'

print('Pytorch version: ', torch.__version__)
if USE_CUDA:
    print('Using GPU, Pytorch linked cuda version: ', torch.version.cuda)
else:
    print('Using CPU, no GPU is fund in pytorch')
