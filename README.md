﻿# Visual-Object-Detection

In this repository, I use OpenCV and PyTorch to ship object detection algorithms for the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset.

That dataset is a quick and simple one to get started with learning & trying out object detection & image recognition.

The source for the data can be simply collected from PyTorch's `torchvision.datasets`.

## Contents

- In `Pre-Trained-Model-Pascal-VOC.ipynb`, I used a pretrained model from Torch Vision that is a Faster R-CNN (Regional Convolutional Neural Network). It uses a ResNet-50 with a Feature Pyramid Network (FPN) feature extractor. Hence the name `fasterrcnn_resnet50_fpn`.
    - I also took advantage of the fact that I was using a GPU on Google Colab, which happened to install Nvidia's CuFFT for GPU-accelerated Fast-Fourier Transform during my library setup.
    - Given that FFT has become very useful for image analysis, I tried out OpenCV's basic FFT, as well as the CUDA FFT implementation. Time metrics easily showed that CuFFT achieved a 6x speedup!
