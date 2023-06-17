---
layout: post
title: DenseNet Implementation
date: 2023-06-02 15:53:00-0400
description: Understanding and implementing Dense Networks
categories: computer-vision implementation
giscus_comments: false
related_posts: true
toc:
  sidebar: right
---
Link to [paper](https://arxiv.org/abs/1608.06993)
Link to [my code](https://github.com/boosungkim/milestone-cnn-model-implementations)

DenseNets are basically ResNets but with residual connections between every single layer.

<img src="/assets/img/blogs/2023-06-02-densenet-implementation/densenet.png"  width="600">
<!-- ![image](/assets/img/blogs/2023-06-02-densenet-implementation/densenet.png) -->
*Figure 1: DenseNet*

The idea is to concatenate the feature maps of all the preceding layers with the result of the current layer. The authors of "Densely Connected Convolutional Networks" argue that the element-wise summation used in Residual Networks may impede the flow of information.

By concatenating the feature maps, the authors improves the information flow as each layer would have direct information from all previous layers.

$$x_l = H_l([x_0, x_1, \dots, x_{l-1}])$$

What this equation basically means is that instead of each layer taking in only the previous output, like VGG, taking in the previous output and the output of the block before, like ResNet, each layer takes in the concatenation of all previous layers in the Dense block.

## Growth rate
So what is the growth rate in DenseNets? The growth rate, as its name suggests, is the growth rate of the number of channels for each dense layer. For instance, if we start off with 64 layers and the growth rate is 32, the convolutional sequence in the first layer will create an output of 32 channels. The dense layer will conclude by concatenating the previous outputs, giving us 96 channels. The next layer will then have 128 layers, and so on.

The authors of the paper emperically prove that the dense network works sufficiently well with relatively small growth rates, like $$k=12$$.

## Bottleneck layers in DenseNet
With the continuous increase in the number of channels, you may understand the importance of using bottlenecks if you read my previous blog post on [ResNet](/_posts/2023-06-01-resnet34-implementation.md).

To recap, bottleneck layers decrease the number of channels using 1 by 1 convolutions for more efficient computation.

By employing 1 by 1 convolutions, the bottleneck layers compress the information carried by the feature maps, reducing the computational load while still preserving essential features.

Unlike ResNet, DenseNets utilize just one 1 by 1 convolutions per layer to reduce the number of channels.

## Transition layers
Bottleneck layers alone are not enough to improve the model compactness. This is where the authors introduce compression by Transition layers.

Essentially, the transition layer reduces the number of channels by a factor of $$\theta$$, which the paper sets as 0.5 for DenseNet-C. Hence, DenseNet-C halves the number of channels every transition layer.

## Implementation detail
We now have the building blocks for DesneNet.

### Dense layer
```python
class DenseLayer(nn.Module):
    def __init__(self, input_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.growth_rate = growth_rate

        self.layer = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels, out_channels=4*input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4*input_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*input_channels, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def forward(self,x):
        return self._forward_implementation(x)
    
    def _forward_implementation(self,x):
        z = self.layer(x)
        z = torch.cat([z, x], 1)
        return z
```
The difference here from the Residual block code is that we use `torch.cat([z,x],1)` to concatenate the previous results. `z` is the result of the convolutions, which will return `growth_rate` number of channels. `x` is the result of the previous layer's output.

### Transition layer
```python
class TransitionBlock(nn.Module):
    def __init__(self, input_channels):
        super(TransitionBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = input_channels // 2

        self.layer = nn.Sequential(
            nn.BatchNorm2d(self.input_channels),
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
    
    def forward(self,x):
        return self.layer(x)
    
    def output_channels_num(self):
        return self.output_channels
```
Even simpler! We just use a combination of BatchNorm, 1 by 1 convolution, and AvgPool to reduce the number of channels by a factor of 0.5 and the width and height by 0.5.

I just set $$\theta$$ to be 0.5 as the paper does, but you can adjust it to be some other value.

## Conclusion
Despite having a similar depth and the number of parameters as ResNet, my variation of DenseNet produced an improvement in both training and testing accuracies. My densenet produced a training accuracy of 96.6% and testing accuracy of 88.7%.

The higher training accuracy before plateuing proves that densely connected layers is an improvement over shortcut connections from ResNet. The direct information flow between layers in DenseNet through concatenation has proven to be beneficial for learning complex representations and improving model performance.