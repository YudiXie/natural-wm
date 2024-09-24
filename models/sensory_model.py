import math

import torch
import torch.nn as nn
import torch.nn.functional as F

fl = math.floor


class LeNet(nn.Module):
    def __init__(self, image_size, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.last_map_x = fl((fl((image_size[1]-4)/2)-4)/2)
        self.last_map_y = fl((fl((image_size[2]-4)/2)-4)/2)

        self.linear1 = nn.Linear(16 * self.last_map_x * self.last_map_y, 120)
        self.linear2 = nn.Linear(120, 84)
        self.out_layer = nn.Linear(84, num_classes)

    def forward(self, inp):
        x = self.pool1(F.relu(self.conv1(inp)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.last_map_x * self.last_map_y)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        otp = self.out_layer(x)
        return otp

    def set_mode(self, mod):
        pass


def get_out_size(
        d_in: int, 
        kernel_size: int, 
        stride: int, 
        padding: int, 
        dilation: int
        ) -> int:
    return math.floor(((d_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)


def get_new_3x3insize2d(insize2d: tuple, stride: int) -> tuple:
    """
    insize2d: input spatial dimensions, form (H, W)
    stride: stride of the convolutional layer
    """
    return tuple([get_out_size(d, 3, stride, 1, 1) for d in insize2d])


def get_new_1x1insize2d(insize2d: tuple, stride: int) -> tuple:
    """
    insize2d: input spatial dimensions, form (H, W)
    stride: stride of the convolutional layer
    """
    return tuple([get_out_size(d, 1, stride, 0, 1) for d in insize2d])


class InstanceNorm2d_affine(nn.Module):

    def __init__(self, C, h, w, **kwargs):
        super(InstanceNorm2d_affine, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(C, affine=False, **kwargs)
        self.weight = nn.Parameter(torch.ones(C, 1, 1))
        self.bias = nn.Parameter(torch.zeros(C, 1, 1))
        self.spatial_weight = nn.Parameter(torch.ones(1, h, w))
        self.spatial_bias = nn.Parameter(torch.zeros(1, h, w))

    def forward(self, x):
        x = self.instance_norm(x)
        x = x * self.weight * self.spatial_weight + self.bias + self.spatial_bias
        return x

def get_normlayer(norm_layer, num_channel, size2d):
    """
    size2d: spatial size of the layer to be normalized, form (H, W)
    """
    if norm_layer is None or norm_layer == 'batchnorm':
        norm_layer = nn.BatchNorm2d(num_channel)
    elif norm_layer == 'instancenorm':
        norm_layer = nn.InstanceNorm2d(num_channel, affine=False)
    elif norm_layer == 'instancenorm_affine':
        norm_layer = InstanceNorm2d_affine(num_channel, size2d[0], size2d[1])
    elif norm_layer == 'layernorm':
        norm_layer = nn.LayerNorm([num_channel, size2d[0], size2d[1]], elementwise_affine=False)
    elif norm_layer == 'groupnorm':
        num_c_per_group = 8
        if num_channel > num_c_per_group:
            assert num_channel % num_c_per_group == 0, "num_channel must be divisible"
        norm_layer = nn.GroupNorm(num_groups=max(1, num_channel // num_c_per_group), 
                                  num_channels=num_channel)
    elif norm_layer == 'none':
        norm_layer = nn.Identity()
    else:
        raise NotImplementedError
    return norm_layer


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                     bias=bias)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, insize2d, 
                 stride=1, downsample=None, norm_layer=None):
        """
        insize2d: input spatial dimension of this block, form (H, W)
        """
        super(BasicBlock, self).__init__()
        use_bias = (norm_layer == 'none')  # use bias in conv only if not using norm layer
        
        if stride != 1:
            insize2d = get_new_3x3insize2d(insize2d, stride)
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, bias=use_bias)
        self.norm1 = get_normlayer(norm_layer, planes, insize2d)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=use_bias)
        self.norm2 = get_normlayer(norm_layer, planes, insize2d)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# adapted from torchvision.models.resnet
# implemented using ResNet for CIFAR-10 setting in:
# Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition. arXiv:1512.03385
# 3 layers, each layer has layers[i] blocks, each block has 2 conv layers and one skip connection.
class ResNet(nn.Module):
    def __init__(self, layers, image_size, num_classes=10, zero_init_residual=False,
                 norm_layer=None, width=64, spatial_average=True):
        super(ResNet, self).__init__()

        self._norm_layer = norm_layer
        self.use_bias = (norm_layer == 'none')  # use bias in conv only if not using norm layer
        assert width % 4 == 0, "width must be divisible by 4"
        self.inplanes = width // 4  # width of the first conv layer
        self.insize2d = (image_size[1], image_size[2])  # image size of the input (H, W)

        self.conv1 = nn.Conv2d(image_size[0], self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=self.use_bias)
        self.norm1 = get_normlayer(norm_layer, self.inplanes, self.insize2d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(width // 4, layers[0])
        self.layer2 = self._make_layer(width // 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width, layers[2], stride=2)

        if spatial_average:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            out_size = 1
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
            out_size = 16
        
        self.out_layer = nn.Linear(width * out_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, 
                                nn.LayerNorm, nn.InstanceNorm2d)):
                # instance norm dose not have learnable weight and bias by default
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last normalization in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)

    def _make_layer(self, planes, blocks, stride=1):
        """
        planes: number of output channels of this layer
        blocks: number of res-blocks in this layer
        stride: if stride != 1, the first block will shrink the spatial size
        """
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes:
            new_insize2d = get_new_1x1insize2d(self.insize2d, stride)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride, bias=self.use_bias),
                get_normlayer(norm_layer, planes, new_insize2d),
            )

        layers = []
        # shrinking of image and expansion of channels happens in the first block
        layers.append(BasicBlock(self.inplanes, planes, self.insize2d, 
                                 stride, downsample, norm_layer))
        self.inplanes = planes
        self.insize2d = get_new_3x3insize2d(self.insize2d, stride)

        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, self.insize2d,
                                     norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.out_layer(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def set_mode(self, mod):
        pass


class MultiscaleResNet(ResNet):
    """
    Same as resnet, except output is the concatenation of 3 blocks
    """
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        out1 = torch.flatten(self.avgpool(out1), 1)
        out2 = torch.flatten(self.avgpool(out2), 1)
        out3 = torch.flatten(self.avgpool(out3), 1)

        out3 = self.out_layer(out3)
        x = torch.cat([out1, out2, out3], dim=1)

        return x


class ResNetSimCLR(nn.Module):
    def __init__(self, config, image_size):
        super(ResNetSimCLR, self).__init__()
        self.out_dim = 128  # feature dimension (default: 128)

        self.backbone = ResNet(
                            [3, 3, 3], image_size, 
                            norm_layer=config.cnn_norm,
                            width=config.cnn_width,
                            num_classes=self.out_dim,
                        )
        dim_mlp = self.backbone.out_layer.in_features

        # add mlp projection head
        self.backbone.out_layer = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                nn.ReLU(), self.backbone.out_layer)

    def forward(self, inp):
        return self.backbone(inp)

    def set_mode(self, mod):
        pass


# M5 convolutional network for auditory classification
# adapted from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio_tutorial.html
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze(dim=1)

    def set_mode(self, mod):
        pass