import os.path as osp
import torch
import torchvision.models as tvmodels

from configs.config_global import DATA_DIR


if __name__ == "__main__":
    # set the torch hub directory
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))
    net = tvmodels.resnet18(weights=tvmodels.ResNet18_Weights.IMAGENET1K_V1)
    net = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.IMAGENET1K_V2)
    net = tvmodels.alexnet(weights=tvmodels.AlexNet_Weights.IMAGENET1K_V1)
    net = tvmodels.vit_b_16(weights=tvmodels.ViT_B_16_Weights.IMAGENET1K_V1)