import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SegResNet, UNet, AHNet, UNETR

SEGRESNET = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
)

UNET = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    dropout=0.2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

AHNET = AHNet(
    layers=(3, 4, 6, 3),
    spatial_dims=3,
    psp_block_num=3,
    in_channels=4,
    out_channels=3,
    upsample_mode='transpose',
    pretrained=False,
)

UNTR = UNETR(
    in_channels=4,
    out_channels=3,
    dropout_rate=0.2,
    img_size=(240, 240, 160),
    spatial_dims=3,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12
)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))