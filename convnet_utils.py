import torch
import torch.nn as nn
from diversebranchblock import DiverseBranchBlock
from acb import ACBlock
from dbb_transforms import transI_fusebn

CONV_BN_IMPL = 'base'

DEPLOY_FLAG = False

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                      stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    if CONV_BN_IMPL == 'base' or kernel_size == 1 or kernel_size >= 7:
        blk_type = ConvBN
    elif CONV_BN_IMPL == 'ACB':
        blk_type = ACBlock
    else:
        blk_type = DiverseBranchBlock
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG)

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    if CONV_BN_IMPL == 'base' or kernel_size == 1 or kernel_size >= 7:
        blk_type = ConvBN
    elif CONV_BN_IMPL == 'ACB':
        blk_type = ACBlock
    else:
        blk_type = DiverseBranchBlock
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU())


def switch_conv_bn_impl(block_type):
    assert block_type in ['base', 'DBB', 'ACB']
    global CONV_BN_IMPL
    CONV_BN_IMPL = block_type

def switch_deploy_flag(deploy):
    global DEPLOY_FLAG
    DEPLOY_FLAG = deploy
    print('deploy flag: ', DEPLOY_FLAG)


def build_model(arch):
    if arch == 'ResNet-18':
        from resnet import create_Res18
        model = create_Res18()
    elif arch == 'ResNet-50':
        from resnet import create_Res50
        model = create_Res50()
    elif arch == 'MobileNet':
        from mobilenet import create_MobileNet
        model = create_MobileNet()
    else:
        raise ValueError('TODO')
    return model