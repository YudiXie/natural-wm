import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'FiLM_att',
    'Elementwise_att',
    'Spatial_att',
    'Identity_att',
    'CBAM_att',
    'CBAM_Spatial_att',
    'Feature_att'
]

class FiLM_att(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.film_dim = dim2
        self.linear = nn.Linear(dim1, dim2 * 2)

    def get_film(self, params):
        params = self.linear(params)
        gammas, betas = params[:, :self.film_dim], params[:, self.film_dim:]
        return gammas, betas

    def forward(self, features, params):
        gammas, betas = self.get_film(params)
        self.features = gammas.detach().cpu(), betas.detach().cpu()
        gammas = gammas.unsqueeze(2).unsqueeze(3)
        betas = betas.unsqueeze(2).unsqueeze(3)
        return (features * gammas) + betas

class Elementwise_att(nn.Module):
    def __init__(self, dim1, num_channels, img_width, img_height):
        super().__init__()
        self.num_channels = num_channels
        self.width = img_width
        self.height = img_height
        self.num_pixels = num_channels * img_width * img_height
        self.linear = nn.Linear(dim1, self.num_pixels + num_channels)

    def get_att(self, params):
        params = self.linear(params)
        factors = params[:, :self.num_pixels]
        bias = params[:, self.num_pixels:]
        factors = factors.reshape(-1, self.num_channels, self.width, self.height)
        bias = bias.reshape(-1, self.num_channels, 1, 1)
        return factors, bias

    def forward(self, features, params):
        factors, bias = self.get_att(params)
        self.features = factors.detach().cpu(), bias.detach().cpu()
        return (features * factors) + bias

class Spatial_att(nn.Module):

    def __init__(self, in_dim, num_channels, img_width, img_height):
        super().__init__()
        self.num_channels = num_channels
        self.width = img_width
        self.height = img_height

        intermediate_size = in_dim + num_channels
        self.linear_param = nn.Linear(in_dim, intermediate_size)
        self.linear_feature = nn.Linear(num_channels, intermediate_size)
        self.out_layer = nn.Linear(intermediate_size, 1)

    def forward(self, features: torch.Tensor, params: torch.Tensor):
        params = self.linear_param(params) # batchsize * inter_size
        x = features.permute(0, 2, 3, 1) # put the channel dimension to the last
        x = self.linear_feature(x) # batchsize * h * w * inter_size
        x = F.relu(x + params.unsqueeze(1).unsqueeze(1))
        x = self.out_layer(x).squeeze().unsqueeze(1) # attention weights: batchsize * 1 * h * w 
        self.features = x.squeeze().detach().cpu()
        return features * x

class Identity_att(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, features, params):
        self.features = None
        return features

class CBAM_att(nn.Module):

    def __init__(self, in_dim, num_channels, img_width, img_height, activation='sigmoid'):
        super().__init__()
        self.num_channels = num_channels
        self.width = img_width
        self.height = img_height
        self.num_pixels = img_width * img_height

        self.linear = nn.Linear(in_dim, self.num_pixels + num_channels)
        self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.Identity()

    def get_spa(self, params):
        params = self.activation(self.linear(params))
        spatial_att = params[:, :self.num_pixels]
        channel_att = params[:, self.num_pixels:]

        spatial_att = spatial_att.reshape(-1, 1, self.width, self.height)
        channel_att = channel_att.reshape(-1, self.num_channels, 1, 1)
        return spatial_att, channel_att

    def forward(self, features, params):
        spatial_att, channel_att = self.get_spa(params)
        self.features = spatial_att.squeeze().detach().cpu(), channel_att.squeeze().detach().cpu()
        return features * channel_att * spatial_att 

class CBAM_Spatial_att(nn.Module):

    def __init__(self, in_dim, num_channels, img_width, img_height, activation='sigmoid'):
        super().__init__()
        self.num_channels = num_channels
        self.width = img_width
        self.height = img_height
        self.num_pixels = img_width * img_height

        self.linear = nn.Linear(in_dim, self.num_pixels)
        self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.Identity()

    def get_spa(self, params):
        params = self.activation(self.linear(params))
        spatial_att = params

        spatial_att = spatial_att.reshape(-1, 1, self.width, self.height)
        return spatial_att

    def forward(self, features, params):
        spatial_att = self.get_spa(params)
        self.features = spatial_att.squeeze().detach().cpu()
        return features * spatial_att 

class Feature_att(nn.Module):

    def __init__(self, in_dim, num_channels, activation='sigmoid'):
        super().__init__()
        self.num_channels = num_channels
        self.linear = nn.Linear(in_dim, num_channels)
        self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.Identity()

    def get_spa(self, params):
        params = self.activation(self.linear(params))
        channel_att = params
        channel_att = channel_att.reshape(-1, self.num_channels, 1, 1)
        return channel_att

    def forward(self, features, params):
        channel_att = self.get_spa(params)
        self.features = channel_att.squeeze().detach().cpu()
        return features * channel_att