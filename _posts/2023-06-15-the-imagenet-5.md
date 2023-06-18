---
layout: post
title: I implemented 5 CNN papers. Here's what I learned.
date: 2023-06-15 15:53:00-0400
description: 5 things I learnt from implementing 5 Computer Vision papers
categories: computer-vision implementation
giscus_comments: false
related_posts: true
toc:
  sidebar: right
---
[The Imagenet 5](https://github.com/boosungkim/milestone-cnn-model-implementations)

*An accompanying Converge2Diverge YouTube video is in the works.*

As of this post, I have implemented 5 milestone CNN papers (6 really, if you include my implementation of MobileNet's inverted residual block). I thinking about calling this short series of implementations "The ImageNet 5" (since all these models are based on ImageNet).

From this short experience, I truly believe that implementing papers and utilizing the code for other projects is the best way to learn deep learning. I learnt a lot, which can be summarized in 5 key points:

## 1. Break down the model
When I first implemented the VGG network and Residual Networks, I coded everything into one huge chunk. If you look at the ealiest GitHub versions of my code, you will be able to find it.

While the code may run, it is significantly harder to debug or to come back to later. If you read my post on [EfficientNet](/_posts/2023-06-15-efficientnet-implementation.md), you saw how we had to reuse the SE-Net code.

Additionally, if you run `summary()` from torchinfo, you can see that the function groups layers together by blocks or sequences. Organizing the dozens of of layers in the model into `features`, `classifier`, or other categories can be immensely helpful in reading the summary later. The official Pytorch implementations of the models do this as well, and when you utilize transfer learning, you can set which blocks of code to training mode easily due to this classification.

Finally, breaking down the model to smaller parts and implementing them first can help your understanding as well. Instead of overwhelming yourself with implementing everything, focus on implementing separate parts of the model.

## 2. You need to know the background knowledge
I particularly felt this when implementing EfficientNet. Trying to implement it without proper knowledge of MobileNetV2 (its basis) was a complete nightmare, and I had to reference many explanations and videos to even code a running model.

When reading a paper, it includes references to previous works it is built on. Don't be stubborn and actually read the references as well.

## 3. Take notes on the paper

<img src="/assets/img/blogs/2023/2023-06-15-the-imagenet-5/notes-densenet.jpeg"  width="600" height="300">

*My notes on the DenseNet paper*

<img src="/assets/img/blogs/2023/2023-06-15-the-imagenet-5/notes-resnet.jpeg"  width="600">

*My notes on the ResNet paper*

As mentioned in a previous blog, the parameters and hyperparameters of a model is often embedded in several different places in the paper. It's like going on an Easter egg hunt for these clues.

Thanks to my prior research experience, I knew how to read papers on a high level: read the abstract -> conclusion -> figures/tables -> introduction -> main content. But through these implementations, I've gotten better at hunting down these little details.

Taking notes and writing down these details are very important, especially when you are trying to read several papers at once and start getting overwhelmed.

## 4. Have a set environment
When you are testing these models, it can be very time-consuming to create a whole new environment. It also helps to use the same dataset with the same augmentations to benchmark your models. I ended up reusing the data augmentation and training/testing codes for all my models.

## 5. Hyperparameter tuning takes all day
The models I implemented were made for ImageNet, not CIFAR10. Thus, I had to finetune the hyperparameters to get close to the results on the paper. Even then, I was not able to fully replicate the results.

These models get more complicated as time goes on. We aren't dealing with MNIST or Cats vs Dogs anymore. Each time I tweak a hyperparameter, I have to let the model run for hundreds of epoch, which can take hours. I've learnt to let the model run while I take care of other work or even while I am sleeping. It is important to get more efficient in running these models.

## What now?
### More experimentation
So far, all my implementations focused on just the coding part with a little bit of finetuning. If my results matched the general trend of the model, I moved onto the next model. That's not what a true researcher does.

A true researcher would try to replicate the different experiments done in each paper. While I will continue with my studies in other domains, I plan on coming back to some of these papers and properly replicating the results. I need to learn to plot loss and accuracy and compare these models against each other under the exact same circumstances.

### Newer papers
So far, I only worked with older papers, the latest one being published in 2020 (which is pretty recent I guess). These older papers have more articles and explanations online that I was able to reference when I got stuck.

For instance, I was able to read an online article explaining the concept of growth rates in DenseNet, which was how I figured out that growth rates are just the number of channels the convolutional step is supposed to output.

As my skills and knowledge improve, I hope to be able to understand newer state-of-the-art papers from top conferences. My current goal is to be able to read and implement a 2023 NeurIPS paper this winter break.

### Other domain studies
There are so many other areas outside of just image classification. The closest areas being Object Segmentation and Object Detection. Not to mention there is a whole world of Natural Language Processing waiting as well.

Perhaps I could do a Segmentation 5?

On a final note, I would like to thank my friend Daesung (Dan) Kim for his continuous help and support so far.