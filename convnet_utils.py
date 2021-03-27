import torch
import torch.nn as nn
from diversebranchblock import DiverseBranchBlock
from acb import ACBlock

CONV_BN_IMPL = 'base'

DEPLOY_FLAG = False

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    if CONV_BN_IMPL == 'base' or kernel_size == 1 or kernel_size >= 7:
        se = nn.Sequential()
        se.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
        se.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
        return se
    elif CONV_BN_IMPL == 'ACB':
        return ACBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG)
    else:
        return DiverseBranchBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG)


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    if CONV_BN_IMPL == 'base' or kernel_size == 1 or kernel_size >= 7:
        se = nn.Sequential()
        se.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
        se.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
        se.add_module('relu', nn.ReLU())
        return se
    elif CONV_BN_IMPL == 'ACB':
        return ACBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU())
    else:
        return DiverseBranchBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU())

def switch_conv_bn_impl(block_type):
    assert block_type in ['base', 'DBB', 'ACB']
    global CONV_BN_IMPL
    CONV_BN_IMPL = block_type

def switch_deploy_flag(deploy):
    global DEPLOY_FLAG
    DEPLOY_FLAG = deploy