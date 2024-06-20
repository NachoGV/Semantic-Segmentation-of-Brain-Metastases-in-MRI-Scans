import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, UNet, AHNet, UNETR

ENSEMBLE_SEGRESNET = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=3,
    out_channels=3,
    dropout_prob=0.2,
)

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

class CNN3D_MODEL(nn.Module):
    def __init__(self):
        super(CNN3D_MODEL, self).__init__()
        # First Layer - Input
        self.conv1_1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1_4 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.relu1 = nn.ReLU()
        # Second Layer - CNN Block 1
        self.conv2 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        # Third Layer - CNN Block 2
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU()
        # Fourth Layer - Output
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Input & Concat
        x1 = self.conv1_1(x[0])
        x2 = self.conv1_2(x[1])
        x3 = self.conv1_3(x[2])
        x4 = self.conv1_4(x[3])
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # Layers
        x = self.relu1(self.bn1(x))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        # Output
        x = self.conv4(x)
        
        return x
        
class CNN3D_CHANNEL(nn.Module):
    def __init__(self):
        super(CNN3D_CHANNEL, self).__init__()
        # First 3D convolutional layer for each input
        self.conv1_1 = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        # Separate batch norm for each branch
        self.bn1_1 = nn.BatchNorm3d(32)
        self.bn1_2 = nn.BatchNorm3d(32)
        self.bn1_3 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()
        # Second 3D convolutional layer
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        # Third 3D convolutional layer
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        # Final 3D convolutional layers (one for each output channel)
        self.conv4_1 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x is a list of three tensors, each with shape (1, 4, 240, 240, 160)
        
        # Process each input tensor separately
        x1 = self.relu(self.bn1_1(self.conv1_1(x[0])))
        x1 = self.relu(self.bn2(self.conv2(x1)))
        x1 = self.relu(self.bn3(self.conv3(x1)))
        out1 = self.conv4_1(x1)  # Output for the first channel
        
        x2 = self.relu(self.bn1_2(self.conv1_2(x[1])))
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x2 = self.relu(self.bn3(self.conv3(x2)))
        out2 = self.conv4_2(x2)  # Output for the second channel
        
        x3 = self.relu(self.bn1_3(self.conv1_3(x[2])))
        x3 = self.relu(self.bn2(self.conv2(x3)))
        x3 = self.relu(self.bn3(self.conv3(x3)))
        out3 = self.conv4_3(x3)  # Output for the third channel
        
        # Concatenate the three output channels
        output = torch.cat([out1, out2, out3], dim=1)
        
        return output  # Shape: (1, 3, 240, 240, 160)