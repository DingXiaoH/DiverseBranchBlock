# Diverse Branch Block: Building a Convolution as an Inception-like Unit (PyTorch) (CVPR-2021)

DBB is a powerful ConvNet building block to replace regular conv. It improves the performance **without any extra inference-time costs**. This repo contains the code for building DBB and converting it into a single conv. You can also get the equivalent kernel and bias in a differentiable way at any time (get_equivalent_kernel_bias in diversebranchblock.py). This may help training-based pruning or quantization.

This is the PyTorch implementation. The MegEngine version is at https://github.com/megvii-model/DiverseBranchBlock

Another nice implementation by [@zjykzj](https://github.com/zjykzj). Please check [here](https://github.com/DingXiaoH/DiverseBranchBlock/issues/20).

Paper: https://arxiv.org/abs/2103.13425

Update: released the code for building the block, transformations and verification.

Update: a more efficient implementation of BNAndPadLayer

Update: MobileNet, ResNet-18 and ResNet-50 models released. You can download them from Google Drive or Baidu Cloud. For the 1x1-KxK branch of MobileNet, we used internal_channels = 2x input_channels for every depthwise conv. 1x also worked but the accuracy was slightly lower (72.71% v.s. 72.88%). On dense conv like ResNet, we used internal_channels = input_channels, and larger internal_channels seemed useless.

Sometimes I call it ACNet v2 because 'DBB' is two bits larger than 'ACB' in ASCII. (lol)

    @inproceedings{ding2021diverse,
    title={Diverse Branch Block: Building a Convolution as an Inception-like Unit},
    author={Ding, Xiaohan and Zhang, Xiangyu and Han, Jungong and Ding, Guiguang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={10886--10895},
    year={2021}
    }

# Abstract

We propose a universal building block of Convolutional Neural Network (ConvNet) to improve the performance without any inference-time costs. The block is named Diverse Branch Block (DBB), which enhances the representational capacity of a single convolution by combining diverse branches of different scales and complexities to enrich the feature space, including sequences of convolutions, multi-scale convolutions, and average pooling. After training, a DBB can be equivalently converted into a single conv layer for deployment. Unlike the advancements of novel ConvNet architectures, DBB complicates the training-time microstructure while maintaining the macro architecture, so that it can be used as a drop-in replacement for regular conv layers of any architecture. In this way, the model can be trained to reach a higher level of performance and then transformed into the original inference-time structure for inference. DBB improves ConvNets on image classification (up to 1.9% higher top-1 accuracy on ImageNet), object detection and semantic segmentation. 

![image](https://github.com/DingXiaoH/DiverseBranchBlock/blob/main/intro.PNG) 
![image](https://github.com/DingXiaoH/DiverseBranchBlock/blob/main/table1.PNG)
![image](https://github.com/DingXiaoH/DiverseBranchBlock/blob/main/table2.PNG)


# Use our pretrained models

You may download the models reported in the paper (ResNet-18, ResNet-50, MobileNet) from Google Drive (https://drive.google.com/drive/folders/1BPuqY_ktKz8LvHjFK5abD0qy3ESp8v6H?usp=sharing) or Baidu Cloud (https://pan.baidu.com/s/1wPaQnLKyNjF_bEMNRo4z6Q, the access code is "dbbk"). For the ease of transfer learning on other tasks, we provide both training-time and inference-time models. For ResNet-18 as an example, assume IMGNET_PATH is the path to your directory that contains the "train" and "val" directories of ImageNet, you may test the accuracy by running
```
python test.py IMGNET_PATH train ResNet-18_DBB_7101.pth -a ResNet-18 -t DBB
```
Here "train" indicates the training-time structure


# Convert the training-time models into inference-time

You may convert a trained model into the inference-time structure with
```
python convert.py [weights file of the training-time model to load] [path to save] -a [architecture name]
```
For example,
```
python convert.py ResNet-18_DBB_7101.pth ResNet-18_DBB_7101_deploy.pth -a ResNet-18
```
Then you may test the inference-time model by
```
python test.py IMGNET_PATH deploy ResNet-18_DBB_7101_deploy.pth -a ResNet-18 -t DBB
```
Note that the argument "deploy" builds an inference-time model.


# ImageNet training

The multi-processing training script in this repo is based on the [official PyTorch example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) for the simplicity and better readability. The modifications include the model-building part and cosine learning rate scheduler. 
You may train and test like this:
```
python train.py -a ResNet-18 -t DBB --dist-url tcp://127.0.0.1:23333 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --workers 64 IMGNET_PATH
python test.py IMGNET_PATH train model_best.pth.tar -a ResNet-18
```


# Use like this in your own code

Assume your model is like
```
class SomeModel(nn.Module):
    def __init__(self, ...):
        ...
        self.some_conv = nn.Conv2d(...)
        self.some_bn = nn.BatchNorm2d(...)
        ...
        
    def forward(self, inputs):
        out = ...
        out = self.some_bn(self.some_conv(out))
        ...
```
For training, just use DiverseBranchBlock to replace the conv-BN. Then SomeModel will be like
```
class SomeModel(nn.Module):
    def __init__(self, ...):
        ...
        self.some_dbb = DiverseBranchBlock(..., deploy=False)
        ...
        
    def forward(self, inputs):
        out = ...
        out = self.some_dbb(out)
        ...
```
Train the model just like you train the other regular models. Then call **switch_to_deploy** of every DiverseBranchBlock, test, and save.
```
model = SomeModel(...)
train(model)
for m in train_model.modules():
    if hasattr(m, 'switch_to_deploy'):
        m.switch_to_deploy()
test(model)
save(model)
```

# FAQs

**Q**: Is the inference-time model's output the _same_ as the training-time model?

**A**: Yes. You can verify that by
```
python dbb_verify.py
```

**Q**: What is the relationship between DBB and RepVGG?

**A**: RepVGG is a plain architecture, and the RepVGG-style structural re-param is designed for the plain architecture. On a non-plain architecture, a RepVGG block shows no superiority compared to a single 3x3 conv (it improves Res-50 by only 0.03%, as reported in the RepVGG paper). DBB is a universal building block that can be used on numerous architectures.

**Q**: How to quantize a model with DBB?

**A1**: Post-training quantization. After training and conversion, you may quantize the converted model with any post-training quantization method. Then you may insert a BN after the conv converted from a DBB and finetune to recover the accuracy just like you quantize and finetune the other models. This is the recommended solution. Please see the quantization example of [RepVGG](https://github.com/DingXiaoH/RepVGG).

**A2**: Quantization-aware training. During the quantization-aware training, instead of constraining the params in a single kernel (e.g., making every param in {-127, -126, .., 126, 127} for int8) for an ordinary conv, you should constrain the equivalent kernel of a DBB (get_equivalent_kernel_bias()). 

**Q**: I tried to finetune your model with multiple GPUs but got an error. Why are the names of params like "xxxx.weight" in the downloaded weight file but sometimes like "module.xxxx.weight" (shown by nn.Module.named_parameters()) in my model?

**A**: DistributedDataParallel may prefix "module." to the name of params and cause a mismatch when loading weights by name. The simplest solution is to load the weights (model.load_state_dict(...)) before DistributedDataParallel(model). Otherwise, you may insert "module." before the names like this
```
checkpoint = torch.load(...)    # This is just a name-value dict
ckpt = {('module.' + k) : v for k, v in checkpoint.items()}
model.load_state_dict(ckpt)
```
Likewise, if the param names in the checkpoint file start with "module." but those in your model do not, you may strip the names like
```
ckpt = {k.replace('module.', ''):v for k,v in checkpoint.items()}   # strip the names
model.load_state_dict(ckpt)
```
**Q**: So a DBB derives the equivalent KxK kernels before each forwarding to save computations?

**A**: No! More precisely, we do the conversion only once right after training. Then the training-time model can be discarded, and every resultant block is just a KxK conv. We only save and use the resultant model.


## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. (preprint, 2021) **A powerful MLP-style CNN building block**\
[RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition](https://arxiv.org/abs/2105.01883)\
[code](https://github.com/DingXiaoH/RepMLP).

2. (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **83.55%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

3. (preprint, 2020) **State-of-the-art** channel pruning\
[Lossless CNN Channel Pruning via Decoupling Remembering and Forgetting](https://arxiv.org/abs/2007.03260)\
[code](https://github.com/DingXiaoH/ResRep).

4. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

5. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)
