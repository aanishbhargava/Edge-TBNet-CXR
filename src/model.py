import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(in_planes // ratio, 8)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.shared_mlp(self.avg_pool(x)) + self.shared_mlp(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class EdgeTBNet(nn.Module):
    def __init__(self, pretrained=True):
        super(EdgeTBNet, self).__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.stage1 = base.features[0:9]
        self.cbam1 = CBAM(in_planes=48, ratio=16)
        self.stage2 = base.features[9:]
        self.cbam2 = CBAM(in_planes=576, ratio=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(576)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(576, 1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.cbam1(x)
        x = self.stage2(x)
        x = self.cbam2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.bn(x)
        x = self.dropout(x)
        return self.fc(x)
