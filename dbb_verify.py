import torch
import torch.nn as nn
from diversebranchblock import DiverseBranchBlock


if __name__ == '__main__':
    x = torch.randn(1, 32, 56, 56)
    for k in (3, 5):
        for s in (1, 2):
            dbb = DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=k, stride=s, padding=k//2,
                                           groups=2, deploy=False)
            for module in dbb.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    nn.init.uniform_(module.running_mean, 0, 0.1)
                    nn.init.uniform_(module.running_var, 0, 0.1)
                    nn.init.uniform_(module.weight, 0, 0.1)
                    nn.init.uniform_(module.bias, 0, 0.1)
            dbb.eval()
            print(dbb)
            train_y = dbb(x)
            dbb.switch_to_deploy()
            deploy_y = dbb(x)
            print(dbb)
            print('========================== The diff is')
            print(((train_y - deploy_y) ** 2).sum())