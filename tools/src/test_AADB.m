clc
clear
warning off
addpath(genpath('tools'));
vl_setupnn;
imdb = setup_AADB();
testID = find(imdb.images.set == 3);


Size = 384;
% load(fullfile('data','AADB','vgg16',num2str(Size),'warp_fc_g0.75_log-3-4PQR','net-epoch-3.mat'),'net');
% net.layers = net.layers(1:end-1);
% net = vl_simplenn_move(net,'gpu');

netStruct = load(fullfile('data','AVA','resnet50',num2str(Size),'RSD_EMD_','net-epoch-10.mat')) ;
net = dagnn.DagNN.loadobj(netStruct.net) ;
% net.removeLayer('softmax_regression')
p = net.getVarIndex('fc') ;
net.vars(p).precious = 1;
net.move('gpu') ;

for i = 1:numel(testID)
    fprintf('processing image %d/%d\n',i,numel(testID));
    imgPath = fullfile(imdb.meta.base_dir,'images',imdb.meta.imageList{testID(i)});
    img = vl_imreadjpeg({imgPath},'numThreads',1,'resize',[Size,Size],'SubtractAverage',net.meta.normalization.averageImage);
    if size(img{1},3) == 1
        img{1} = cat(3,img{1},img{1},img{1});
    end
%     res = vl_simplenn_modified(net,gpuArray(img{1}),[],[],'mode','test');
%     pred_score(i) = imdb.anchors' * squeeze(gather(vl_nnsoftmax(res(end).x)));
    
    net.mode = 'test' ;
    inputs = {'data', gpuArray(img{1})} ;
    net.eval(inputs) ;
    pred_score(i) = [1:10] * squeeze(gather(vl_nnsoftmax(net.vars(p).value)));
%     pred_score(i) = squeeze(gather(vl_nnsigmoid(net.vars(p).value))); 
end

% Size = 384;
% load(fullfile('data','AADB','vgg16',num2str(Size),'warp_bcnn_g0.75_log-3-4PQR','net-epoch-6.mat'),'net');
% net.layers = net.layers(1:end-1);
% net = vl_simplenn_move(net,'gpu');
% 
% for i = 1:numel(testID)
%     fprintf('processing image %d/%d\n',i,numel(testID));
%     imgPath = fullfile(imdb.meta.base_dir,'images',imdb.meta.imageList{testID(i)});
%     img = vl_imreadjpeg({imgPath},'numThreads',1,'resize',[Size,Size],'SubtractAverage',net.meta.normalization.averageImage);
%     if size(img{1},3) == 1
%         img{1} = cat(3,img{1},img{1},img{1});
%     end
%     res = vl_simplenn_modified(net,gpuArray(img{1}),[],[],'mode','test');
%     pred_score384(i) = imdb.anchors' * squeeze(gather(vl_nnsoftmax(res(end).x)));
% end

% Size = 512;
% load(fullfile('data','AADB','vgg16',num2str(Size),'warp_fc_g0.25_log-3-4PQR','net-epoch-10.mat'),'net');
% net.layers = net.layers(1:end-1);
% net = vl_simplenn_move(net,'gpu');
% 
% for i = 1:numel(testID)
%     fprintf('processing image %d/%d\n',i,numel(testID));
%     imgPath = fullfile(imdb.meta.base_dir,'images',imdb.meta.imageList{testID(i)});
%     img = vl_imreadjpeg({imgPath},'numThreads',1,'resize',[Size,Size],'SubtractAverage',net.meta.normalization.averageImage);
%     if size(img{1},3) == 1
%         img{1} = cat(3,img{1},img{1},img{1});
%     end
%     res = vl_simplenn_modified(net,gpuArray(img{1}),[],[],'mode','test');
%     pred_score512(i) = imdb.anchors' * squeeze(gather(vl_nnsoftmax(res(end).x)));
% end

% pred_score = pred_score384;
% pred_score = pred_score256 + pred_score384 + pred_score512;
groundtruth_score = imdb.images.score(testID);
srcc = corr(pred_score',groundtruth_score','type','Spearman');

