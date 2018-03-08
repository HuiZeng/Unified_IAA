%  please replace "tools/matconvnet/matlab/+dagnn/Loss.m" with "tools/src/Loss.m" before run ResNet

% clc;
clear;
warning off
addpath(genpath('tools'));
vl_setupnn;

testDatabase = 'AVA'; % AADB
Model = 'resnet50';
epoch = 10;
fixImgSize = 384;

lossType = 'CE'; % MSE, Huber, CE
labelType = 'qualityScore'; % Gauss_OV Gauss_MV RSD binaryLabel qualityScore

trainModel('Model',Model,'database',testDatabase,'lossType',lossType,...
           'labelType',labelType,'epoch',epoch,'fixImgSize',fixImgSize);




