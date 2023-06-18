---
layout: post
title: SE-Net Implementation
date: 2023-06-05 15:53:00-0400
description: Understanding and implementing SE-Networks
categories: computer-vision implementation
giscus_comments: false
related_posts: true
toc:
  sidebar: right
---
Link to [paper](https://arxiv.org/pdf/1709.01507.pdf)
Link to [my code](https://github.com/boosungkim/milestone-cnn-model-implementations)

Squeeze-and-Excitation Networks, SE-Nets for short, are actually convolutional blocks that can be added to other models, like ResNet or VGG.

![image](/assets/img/blogs/2023/2023-06-05-senet-implementation/senet.png)
*Figure 1: Squeeze and Excitation Block*

The key problem that the authors of the paper wants to address is the problem of implicit and local channel dependencies. They do so by adding the Squeeze-and-Excitation block, which relays channel-wise information.

By incorporating the SE blocks into the model, the network can adaptively recalibrate its feature maps to capture more discriminative information, leading to improved performance.

## Implicit local channel information
![image](/assets/img/blogs/2023/2023-06-05-senet-implementation/convolution.png)
*Figure 2: A normal convolution*

In a normal convolution like the image above, channel dependencies are implicitly included in the outputs of the convolutional layers. In other words, each layer calculates the convolution on all the channels of a local region every step.

Due to the localness of the convolutions, each channel in the output contains implicit channel embeddings tangled with local spatial correlations. To simplify further, each pixel in a channel contains the channel embeddings of the local region on the convolution was calculated on.

## The SE-Net
The SE-Net sets out to resolve this issue by introducing an explicit mechanism to model channel-wise relationships through the "squeeze" and "excitation" layers.

### Squeeze
The network first "squeezes" the outputs of the previous convolutional layer into $$channel\times1\times1$$ shape using Global Average Pool.

### Excitation
The network then performs "excitation" by performing two Fully Connected (FC) layers. The first FC layer reduces the number of channels by applying a reduction ratio. This reduction helps in reducing the computational complexity of the SE block. The second FC layer then expands the number of channels back to the original number. These FC layers capture the channel dependencies and learn channel-wise relationships based on the aggregated information from the squeeze operation.

FYI, because the FC layers are "fully connected," every node is connected with each other. This is how the network captures channel-wise relationships and dependencies.

### Rescale
Finally, the SE-Net rescales the output back to the input dimensions by using the Unflatten operation and channel-wise multiplication. The channel-wise attention weights are then applied by element-wise multiplication, allowing the network to selectively amplify or suppress the channel activations based on their importance.

## SE-Net implementation
### Implementation
```python
class se_block(nn.Module):
    def __init__(self, input_channels, reduction_ratio):
        super(se_block,self).__init__()
        self.sequence = nn.Sequential(
            # 1. Squeeze
            nn.AdaptiveAvgPool2d((1,1)), # output: bxCx1x1
            nn.Flatten(),  # output: bxC
            # 2. Excitation
            nn.Linear(input_channels, input_channels // reduction_ratio, bias=False), # output: bxC/r
            nn.ReLU(), # output: bxC/r
            nn.Linear(input_channels // reduction_ratio, input_channels), # output: bxC
            nn.Sigmoid(), # output: bxC
            nn.Unflatten(1, (input_channels,1,1)) # output: bxCx1x1
        )
        pass

    def forward(self,x):
        z = self.sequence(x)
        # 3. Rescale
        z = x*z # output: bxCxHxW
        return z
```
There is not much to add here, as the code follows the description one-to-one. One thing to add is that I originally used Pytorch's AvgPool2d with manually calculated channel width and height, but Pytorch has the AdaptiveAvgPool2d which handles the dimensions for you.

### Network summary
```python
>>> model = se_block(64, 16)
>>> summary(model, input_size=(1,64,32,32), col_names=["input_size","output_size","num_params"])
```
```
===================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
===================================================================================================================
se_block                                 [1, 64, 32, 32]           [1, 64, 32, 32]           --
├─Sequential: 1-1                        [1, 64, 32, 32]           [1, 64, 1, 1]             --
│    └─AdaptiveAvgPool2d: 2-1            [1, 64, 32, 32]           [1, 64, 1, 1]             --
│    └─Flatten: 2-2                      [1, 64, 1, 1]             [1, 64]                   --
│    └─Linear: 2-3                       [1, 64]                   [1, 4]                    256
│    └─ReLU: 2-4                         [1, 4]                    [1, 4]                    --
│    └─Linear: 2-5                       [1, 4]                    [1, 64]                   320
│    └─Sigmoid: 2-6                      [1, 64]                   [1, 64]                   --
│    └─Unflatten: 2-7                    [1, 64]                   [1, 64, 1, 1]             --
===================================================================================================================
Total params: 576
Trainable params: 576
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 0.00
===================================================================================================================
Input size (MB): 0.26
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.27
===================================================================================================================
```
You may notice that the number of parameters is relatively low due to the use the reduction ratio. Thus, the inclusion of SE-Nets in other models will not significantly impact the number of total parameters.

## SE-ResNet
With the SE-Net coded, it is trivial to add it to our previous models. I decided to test it on ResNet.

```python
class SE_ResidualBlockBottleneck(nn.Module):
    expansion = 4
    def __init__(self, input_channels, in_channels, reduction_ratio, stride=1):
        super(SE_ResidualBlockBottleneck, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels*4)
        )
        self.relu = nn.ReLU()
        self.se_block = SE_block(in_channels*4, reduction_ratio)

        if stride != 1 or input_channels != self.expansion*in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=in_channels*4, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(in_channels*4)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        z = self.block(x)
        z = self.se_block(z)
        z += self.shortcut(x)
        z = self.relu(z)
        return z
```
Notice that the only additions made are `self.se_block = SE_block(in_channels*4, reduction_ratio)` and `z = self.se_block(z)`. The number of channels is `in_channel*4` as that is the result of the residual block (bottleneck).

### Experimentation
When run under the same hyperparameters as ResNet, SE-ResNet produces a training accuracy of 95.7% and a testing accuracy of 88.6%. Not quite as a good as DenseNet, but I only ran SE-ResNet for about 400 epoch, significantly less than my runs for ResNet and DenseNet.

## Conclusion
The results fall in line with the results of the paper. The accuracies were increased and the network converged faster than the original.

While I only tested the SE-Network on ResNet, the squeeze-and-excitation method can be included in many other models as well.

## References:
1. https://towardsdatascience.com/practical-graph-neural-networks-for-molecular-machine-learning-5e6dee7dc003