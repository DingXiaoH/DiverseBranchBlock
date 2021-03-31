import torch.nn as nn
import torch.nn.functional as F
from convnet_utils import conv_bn, conv_bn_relu

def create_stem(channels):
    stem = nn.Sequential()
    stem.add_module('conv1', conv_bn_relu(in_channels=3, out_channels=channels[0], kernel_size=11, stride=4, padding=2))
    stem.add_module('maxpool1', nn.Maxpool2d(kernel_size=3, stride=2))
    stem.add_module('conv2', conv_bn_relu(in_channels=channels[0], out_channels=channels[1], kernel_size=5, padding=2))
    stem.add_module('maxpool2', nn.Maxpool2d(kernel_size=3, stride=2))
    stem.add_module('conv3', conv_bn_relu(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1))
    stem.add_module('conv4', conv_bn_relu(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1))
    stem.add_module('conv5', conv_bn_relu(in_channels=channels[3], out_channels=channels[4], kernel_size=3, padding=1))
    stem.add_module('maxpool3', nn.Maxpool2d(kernel_size=3, stride=2))
    return stem

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        channels = [64, 192, 384, 384, 256]
        self.stem = create_stem(channels)
        self.linear1 = nn.Linear(in_features=channels[4] * 6 * 6, out_features=4096)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        out = self.stem(x)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.linear3(out)
        return out

def create_AlexNet():
    return AlexNet()