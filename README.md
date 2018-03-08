# Introduction

Matlab implementation of "A Unified Probabilistic Formulation for Image Aesthetic Assessment" summitted to TIP.

**How to run the Code**

1. Download the [MatConvNet](http://www.vlfeat.org/matconvnet/) into ``tools`` and compile it according to the guidence therein. 

2. Create a new folder ``data/PretrainedModels`` and download the pre-trained [ResNet50](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat) into this new folder.

3. Create a new folder ``databases``. Download the ``AVA`` dataset into ``databases``. The AVA database can be downloaded following [this guidence](https://about.me/lucamarchesotti).

4. Run ``RunExperiment.m``.
