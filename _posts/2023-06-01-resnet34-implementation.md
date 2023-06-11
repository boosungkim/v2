---
layout: post
title: ResNet34 Model Implementation
date: 2023-06-01 15:53:00-0400
description: Understanding and implementing Residual Networks
categories: computer-vision implementation
giscus_comments: false
related_posts: true
toc:
  sidebar: right
---
Link to [paper](https://arxiv.org/pdf/1512.03385.pdf)
Link to [my code](https://github.com/boosungkim/milestone-cnn-model-implementations)

*This blog is limited to the scope of ResNet34. Additionally commentary on Bottlenecks will be in a future blog post.*

Continuing on from my previous [VGG implementation](./2023-05-31-first-paper-implementation.md), I worked on implementing ResNet next. When my friend said that it is basically VGG, just with skip connections, I figured this would be a piece of cake. Unfortunately, that was not neccessarily the case.

I will mention this again in the Hurdles section, but the main challenge with this paper was figuring out the implementation of the dimensionality reduction in the skip connection.

## What is ResNet?
The researchers behind "Deep Residual Learning for Image Recognition" set out to solve the common 'degradation problem' in deep convolutional networks.

When the depth of a CNN model is increased, it initially shows an improvement in performance but eventually degrades rapidly.

### The common misconception
The common misconception is that this rapid degradation in accuracy is caused by overfitting.

While overfitting due to exploding/vanishing gradients are expected problems in very deep networks, it has been accounted for by nomalized initializations of the dataset and the intermediate Batch Normalization layers.

The degradation is definitely not caused by overfitting, as adding more layers actually causes the *training error* to **increase**. 

So what casues this degradation problem? Even the researchers in the paper are not sure. Their conclusion is that "deep plain nets may have exponentially low convergence rates."

## What is a Residual Learning?
So if we don't know what causes the degradation problem, do we at least know how to prevent it? That's where residual mapping in Residual Learning come into play.

![image](/assets/img/blogs/2023-06-01-resnet34-implementation/residual-block.jpeg)
*Figure 1: A residual block*

$$H(x) := F(x) + x$$

If we let $x$ be the incoming feature, the $F(x)$ is the normal weighted layers that CNNs have (Convolutional, Batch Normalization, ReLU layers). The original $x$ is then added (element-wise addition) to $F(x)$ to produce $H(x)$.

Essentially, the original features are added to the result of the weighted layers, and this whole process is one residual block. The idea is that, in the worst case scenario where $F(x)$ produce a useless tensor filled with $0$s, the identity will be added back in to pass on a useful feature to the next block. Shortcut connections can only help the network, not detriment.

As this is a CNN model, downsampling is necessary. The issue is that the dimensions of $F(x)$ and $x$ would be different after downsampling. In such cases, the $F(x)$ and $W_1x$ are added together, where the square matrix $W_1$ is used to match dimensions.

## Understanding dimensionality reduction
As this is a CNN model, downsampling is necessary. The issue is that the dimensions of $F(x)$ and $x$ would be different after downsampling. In such cases, the $F(x)$ and $W_1x$ are added together, where the square matrix $W_1$ is used to match dimensions.

Below is for simple shortcut connections with elementwise addition.
```python
z = z + identity
```

But what about the dimensionality reductions? Well it's just a sequence of a Conv2d layer and a BatchNorm2d layer, but matching the dimensions was a little confusing at first.
```python
z = z + self.dim_reduction(identity)
...

def skip_connection(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(output_channels)
        )
```

## Conclusion
ResNet definitely felt more complicated to replicate than VGG models. While VGGs are essentially stacking on layer after another, I needed to account for shortcut connections from previous layers to the current layers.