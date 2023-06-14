---
layout: post
title: My first full paper implementation
date: 2023-05-31 15:53:00-0400
description: Implementing the VGG paper from "Very Deep Convolutional Networks for Large-Scale Image Recognition"
categories: computer-vision implementation
giscus_comments: false
related_posts: true
toc:
  sidebar: right
---
Link to [paper](https://arxiv.org/abs/1409.1556)
Link to [my code](https://github.com/boosungkim/milestone-cnn-model-implementations)

I have read several Deep Learning research papers at this point, but I have never fully implemented one from scratch. I decided to finally try with the VGG model from [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).

## VGG Explained
The VGG paper was unique in that it proved the importance of depth in image classification. Unlike its predecessors, VGG utilizes simple filters and a repetitive structure. In return for its simplicity, VGG has a far greater depth, the deepest variation in the paper having 19 layers, which was a lot for its time.

## Hurdles along the way
Implementing VGG from scratch initially introduced a few challenges. While it may seem easy now, converting the architecture represented in the text into Pytorch code was confusing at first.

### 1. Coding and refactoring
At first, I manually wrote the model by simply writing out every single layer in the model class ```__init__()``` function. Obviously, this is not good code.

In the end, I referenced some code online to see how other people organized their code. Others did so by utilizing enumeration, ```for``` loops and helper functions. 

### 2. Finding hyperparameters
Another trouble I ran into was finding the parameters and hyperparameters for the architecture in the code. Thankfully, the VGG paper has a very detailed architecture diagram.

![image](/assets/img/blogs/2023-05-31-first-paper-implementation/vgg-architecture.png)

As can be seen above, the sizes and number of convolutional filters for each layer is quite apparent. However, for instance, I had to read the text to find out that the stride and padding are both 1 pixel.

Not difficult at all, as the paper is very detailed, but some of the more recent papers I tried recently are not as explicit.

### 3. Checking the model structure
Once I had a seemingly functioning model, I needed a way of making sure that recreated the VGG model exactly. There are two methods I used:

1) Passing a tensor through the model

```python
t1 = torch.randn(1,3,224,224)
```
> A tensor representation of an image from ImageNet.

By sending this through the model, I was able to check if the model returns the correct output dimensions.

```python
>>> print(testing(t1).size())
torch.Size([1, 1000])
```

2) Torchinfo

However, this is not enough to ensure that the model architecture is correct. So, I used a Python library called [Torchinfo](https://github.com/TylerYep/torchinfo) to print out the entire architecture. I ran into a few issues with Torchinfo later on, but that is not relevant here.

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
VGGModel                                 [1, 1000]                 --
├─Sequential: 1-1                        [1, 64, 112, 112]         --
│    └─Conv2d: 2-1                       [1, 64, 224, 224]         1,792
│    └─ReLU: 2-2                         [1, 64, 224, 224]         --
│    └─Conv2d: 2-3                       [1, 64, 224, 224]         36,928
│    └─ReLU: 2-4                         [1, 64, 224, 224]         --
│    └─MaxPool2d: 2-5                    [1, 64, 112, 112]         --
├─Sequential: 1-2                        [1, 128, 56, 56]          --
│    └─Conv2d: 2-6                       [1, 128, 112, 112]        73,856
│    └─ReLU: 2-7                         [1, 128, 112, 112]        --
│    └─Conv2d: 2-8                       [1, 128, 112, 112]        147,584
│    └─ReLU: 2-9                         [1, 128, 112, 112]        --
│    └─MaxPool2d: 2-10                   [1, 128, 56, 56]          --
├─Sequential: 1-3                        [1, 256, 28, 28]          --
│    └─Conv2d: 2-11                      [1, 256, 56, 56]          295,168
│    └─ReLU: 2-12                        [1, 256, 56, 56]          --
│    └─Conv2d: 2-13                      [1, 256, 56, 56]          590,080
│    └─ReLU: 2-14                        [1, 256, 56, 56]          --
│    └─Conv2d: 2-15                      [1, 256, 56, 56]          590,080
│    └─ReLU: 2-16                        [1, 256, 56, 56]          --
│    └─Conv2d: 2-17                      [1, 256, 56, 56]          590,080
│    └─ReLU: 2-18                        [1, 256, 56, 56]          --
│    └─MaxPool2d: 2-19                   [1, 256, 28, 28]          --
├─Sequential: 1-4                        [1, 512, 14, 14]          --
│    └─Conv2d: 2-20                      [1, 512, 28, 28]          1,180,160
│    └─ReLU: 2-21                        [1, 512, 28, 28]          --
│    └─Conv2d: 2-22                      [1, 512, 28, 28]          2,359,808
│    └─ReLU: 2-23                        [1, 512, 28, 28]          --
│    └─Conv2d: 2-24                      [1, 512, 28, 28]          2,359,808
│    └─ReLU: 2-25                        [1, 512, 28, 28]          --
│    └─Conv2d: 2-26                      [1, 512, 28, 28]          2,359,808
│    └─ReLU: 2-27                        [1, 512, 28, 28]          --
│    └─MaxPool2d: 2-28                   [1, 512, 14, 14]          --
├─Sequential: 1-5                        [1, 512, 7, 7]            --
│    └─Conv2d: 2-29                      [1, 512, 14, 14]          2,359,808
│    └─ReLU: 2-30                        [1, 512, 14, 14]          --
│    └─Conv2d: 2-31                      [1, 512, 14, 14]          2,359,808
│    └─ReLU: 2-32                        [1, 512, 14, 14]          --
│    └─Conv2d: 2-33                      [1, 512, 14, 14]          2,359,808
│    └─ReLU: 2-34                        [1, 512, 14, 14]          --
│    └─Conv2d: 2-35                      [1, 512, 14, 14]          2,359,808
│    └─ReLU: 2-36                        [1, 512, 14, 14]          --
│    └─MaxPool2d: 2-37                   [1, 512, 7, 7]            --
├─Flatten: 1-6                           [1, 25088]                --
├─Sequential: 1-7                        [1, 1000]                 --
│    └─Linear: 2-38                      [1, 4096]                 102,764,544
│    └─ReLU: 2-39                        [1, 4096]                 --
│    └─Linear: 2-40                      [1, 4096]                 16,781,312
│    └─ReLU: 2-41                        [1, 4096]                 --
│    └─Linear: 2-42                      [1, 1000]                 4,097,000
│    └─Softmax: 2-44                     [1, 1000]                 --
==========================================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 19.65
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 118.89
Params size (MB): 574.67
Estimated Total Size (MB): 694.16
==========================================================================================
```

The example above is my implementation of the VGG19 model. The structure seems to match up with the paper's VGG19. Additionally, the number of parameters of my custom model is roughly equivalent to that of VGG19 (144M parameters)!

![image](/assets/img/blogs/2023-05-31-first-paper-implementation/vgg-param.png)

On a side note, you may wonder why there is no huge difference between the number of parameters in VGG11 (133M) and that of VGG19 (144M). The reason for that is that most of the parameters are located in the Fully Connected layers at the end of the each model. The number of parameters in the Convolutional layers pales in comparison to the number in Fully Connected layers, which is why, even though there is an 8-layer difference between VGG11 and VGG19, the numbers of parameters do not differ that much.

### 4. The dreaded 10% accuracy
When I ran this code on the CIFAR10 dataset, however, I got a 0.1 accuracy! That's basically the same as randomly guessing (since there are 10 possible guesses). No matter how many times I tweaked the hyperparameters, I got the same 0.1.

The VGG model was created to be used on the ImageNet, which contains images of `224x224x3` dimensions. On the other hand, CIFAR10 contains images of `32x32x3` dimensions. The VGG is simple just too deep to learn the CIFAR10 dataset properly.

I resolved this issue by adding Batch Normalization to every block in the architecture. Since Batch Normalization ensures that the activations of each layer have zero mean and unit variance, the gradients are well scaled throughout the network. I improved the performance even further by removing the final few Fully Connected layers.

In the end, with even more hyperparameter tweaking, I was left with a training accuracy of 93.4%, a validation accuracy of 88.7%, and a testing accuracy of 88.2%.

## Conclusion
As someone who had never fully implemented an entire paper before, I struggled with some of the code conversion in the beginning. However, compared to models like ResNet and DenseNet, VGG definitely was the easiest to implement.

I can see why people often recommend beginners to start by implementing papers despite the difficulty - there really is no better way to learn.

What I learnt:
- NN model debugging skills
- Analyzing parameters and hyperparameters from papers
- Adjusting the model to fit the dataset