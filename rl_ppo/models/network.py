import torch
import numpy as np
from torch import nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.ln1 = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU() 
        
        self.conv2 = layer_init(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.ln2 = nn.GroupNorm(1, out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.ln2(out)

        out = self.se(out)

        out += identity
        out = self.act(out)

        return out

class CNNModel(nn.Module):
    def __init__(self, num_channels=256, num_blocks=18):
        nn.Module.__init__(self)
        self.conv_in = nn.Sequential(
            layer_init(nn.Conv2d(45, num_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.GroupNorm(1, num_channels),
            nn.GELU()
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(BasicBlock(num_channels, num_channels))
        self.backbone = nn.Sequential(*blocks)

        self.policy_conv = nn.Sequential(
            layer_init(nn.Conv2d(num_channels, 64, kernel_size=1, stride=1, bias=False)),
            nn.GroupNorm(1, 64),
            nn.GELU()
        )
        self.policy_fc = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 9, 1024)),
            nn.LayerNorm(1024),
            nn.GELU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.LayerNorm(1024),
            nn.GELU(),
            layer_init(nn.Linear(1024, 235), std=0.01) 
        )

        self.value_conv = nn.Sequential(
            layer_init(nn.Conv2d(num_channels, 64, kernel_size=1, stride=1, bias=False)),
            nn.GroupNorm(1, 64),
            nn.GELU()
        )
        self.value_fc = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 9, 1024)),
            nn.LayerNorm(1024),
            nn.GELU(),
            layer_init(nn.Linear(1024, 1), std=1.0)
        )

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        
        x = self.conv_in(obs)
        x = self.backbone(x)
        
        p = self.policy_conv(x)
        logits = self.policy_fc(p)

        v = self.value_conv(x)
        value = self.value_fc(v)
        
        mask = input_dict["action_mask"].float()
        masked_logits = torch.where(mask > 0.5, logits, torch.tensor(-1e9).to(logits.device))
        
        return masked_logits, value