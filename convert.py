import argparse
import os
import torch
from convnet_utils import switch_conv_bn_impl, switch_deploy_flag, build_model

parser = argparse.ArgumentParser(description='DBB Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ResNet-18')

def convert():
    args = parser.parse_args()

    switch_conv_bn_impl('DBB')
    switch_deploy_flag(False)
    train_model = build_model(args.arch)

    if 'hdf5' in args.load:
        from utils import model_load_hdf5
        model_load_hdf5(train_model, args.load)
    elif os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    for m in train_model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()

    torch.save(train_model.state_dict(), args.save)


if __name__ == '__main__':
    convert()