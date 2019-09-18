function [net, info] = trainModel(varargin)


opts.batchNormalization = false ;
opts.network = [];
opts.expDir = [];
opts.networkType = 'simplenn' ;
opts.Model = 'resnet50';
opts.lossType = 'CE';
opts.labelType = 'qualityScore';
opts.database = 'AVA';
opts.epoch = 10;
opts.fixImgSize = 256;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
[opts, varargin] = vl_argparse(opts, varargin);
opts.expDir = fullfile('data',opts.database,opts.Model,num2str(opts.fixImgSize),[opts.labelType,'_',opts.lossType]);
opts.dataDir = fullfile('databases', opts.database);
opts.imdbPath = fullfile('data', 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;

switch opts.Model
  case 'vgg16'
      opts.networkType = 'dagnn' ;
  case {'resnet50','resnet101'} 
      opts.networkType = 'dagnn' ;
end


if exist(opts.imdbPath, 'file')
    load(opts.imdbPath) ;
else
    switch opts.database
        case 'AVA'
            imdb = setup_AVA();
        case 'AADB'
            imdb = setup_AADB();
    end
end

switch opts.labelType
    case {'RSD','Gauss_OV','Gauss_MV'}
        opts.outputDim = numel(imdb.anchors);
    case {'qualityScore','binaryLabel'}
        opts.outputDim = 1;    
end
%% prepare the model and data
net = model_intialization(opts);
dataMean = mean(mean(net.meta.normalization.averageImage,1),2);

imdb.database = opts.database;
imdb.images.data_mean = dataMean;
imdb.images.outputDim = opts.outputDim;
imdb.meta.fixImgSize = opts.fixImgSize;
imdb.meta.lossType = opts.lossType;
imdb.meta.labelType = opts.labelType;
net.meta.normalization.averageImage = dataMean;
net.meta.lossType = opts.lossType;


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn' 
      trainfn = @cnn_train_modified ; 
  case 'dagnn'
      trainfn = @cnn_train_dag_modified;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;


% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels, imgID] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
imgID = [];
fixImgSize = imdb.meta.fixImgSize;
image_path = fullfile(imdb.meta.base_dir,'images',imdb.meta.imageList(batch));
if strcmp(imdb.meta.mode,'train') 
    images = vl_imreadjpeg(image_path,'resize',[fixImgSize,fixImgSize],'pack',...
        'numThreads',4,'SubtractAverage',imdb.images.data_mean,'Flip');
else
    images = vl_imreadjpeg(image_path,'resize',[fixImgSize,fixImgSize],'pack',...
        'numThreads',4,'SubtractAverage',imdb.images.data_mean);
end
images = images{1};

switch imdb.meta.labelType
    case 'Gauss_MV'
        labels = imdb.images.Gauss_MV(:,batch) ;
    case 'Gauss_OV'
        labels = imdb.images.Gauss_OV(:,batch) ;
    case 'RSD'
        labels = imdb.images.RSD(:,batch) ;
    case 'binaryLabel'
        labels = imdb.images.labels(:,batch) ;
    case 'qualityScore'
        labels = imdb.images.score(:,batch);   
end


% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
[images, labels] = getSimpleNNBatch(imdb, batch);
inputs = {'data', images, 'label', labels} ;
