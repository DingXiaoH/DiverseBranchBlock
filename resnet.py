import torch.nn as nn
import torch.nn.functional as F
from convnet_utils import conv_bn, conv_bn_relu

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_bn(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.conv1 = conv_bn_relu(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = conv_bn(in_channels=planes, out_channels=self.expansion * planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = conv_bn(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

        self.conv1 = conv_bn_relu(in_planes, planes, kernel_size=1)
        self.conv2 = conv_bn_relu(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = conv_bn(planes, self.expansion*planes, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1):
        super(ResNet, self).__init__()

        self.in_planes = int(64 * width_multiplier)
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1', conv_bn_relu(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = self._make_stage(block, int(64 * width_multiplier), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, int(128 * width_multiplier), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, int(256 * width_multiplier), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(block, int(512 * width_multiplier), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512*block.expansion*width_multiplier), num_classes)

    def _make_stage(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            if block is Bottleneck:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride))
            else:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def create_Res18():
    return ResNet(BasicBlock, [2,2,2,2], num_classes=1000, width_multiplier=1)


def create_Res50():
    return ResNet(Bottleneck, [3,4,6,3], num_classes=1000, width_multiplier=1)