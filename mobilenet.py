import torch.nn as nn
import torch.nn.functional as F
from convnet_utils import conv_bn_relu
MOBILE_CHANNELS = [32,
                   32, 64,
                   64, 128,
                   128, 128,
                   128, 256,
                   256, 256,
                   256, 512,
                   512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                   512, 1024,
                   1024, 1024]

class MobileV1Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(MobileV1Block, self).__init__()
        self.depthwise = conv_bn_relu(in_channels=in_planes, out_channels=in_planes, kernel_size=3,
                                          stride=stride, padding=1, groups=in_planes)
        self.pointwise = conv_bn_relu(in_channels=in_planes, out_channels=out_planes, kernel_size=1,
                                          stride=1, padding=0)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class MobileV1(nn.Module):

    def __init__(self, num_classes):
        super(MobileV1, self).__init__()
        channels = MOBILE_CHANNELS
        assert len(channels) == 27
        self.conv1 = conv_bn_relu(in_channels=3, out_channels=channels[0], kernel_size=3, stride=2, padding=1)
        blocks = []
        for block_idx in range(13):
            depthwise_channels = int(channels[block_idx * 2 + 1])
            pointwise_channels = int(channels[block_idx * 2 + 2])
            stride = 2 if block_idx in [1, 3, 5, 11] else 1
            blocks.append(MobileV1Block(in_planes=depthwise_channels, out_planes=pointwise_channels, stride=stride))
        self.stem = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stem(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def create_MobileNet():
    return MobileV1(num_classes=1000)