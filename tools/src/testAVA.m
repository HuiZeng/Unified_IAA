clc
clear
warning off
addpath(genpath('tools'));
vl_setupnn;

Size = 384;
% load(fullfile('data','AVA','vgg16',num2str(Size),'gaussDistribution_CE_lr3.0','net-epoch-9.mat'),'net');
% net.layers = net.layers(1:end-1);
% net = vl_simplenn_move(net,'gpu');

netStruct = load(fullfile('data','AVA','resnet50',num2str(Size),'RSD_EMD_','net-epoch-9.mat')) ;
net = dagnn.DagNN.loadobj(netStruct.net) ;
p = net.getVarIndex('fc') ;
net.vars(p).precious = 1;
net.move('gpu') ;


imdb = setup_AVA();
imdb.anchors = [1:10]';
% imdb = setup_AADB();
testID = find(imdb.images.set == 3);
groundtruth_score = imdb.images.score(testID);
RSD = imdb.images.RSD(:,testID);
bs = 64;
for i = 1:bs:numel(testID)
    idx = i:min((i+bs-1),numel(testID));
    fprintf('processing image %d--%d/%d\n',i,(i+bs-1),numel(testID));
    imgPath = fullfile(imdb.meta.base_dir,'images',imdb.meta.imageList(testID(idx)));
    img = vl_imreadjpeg(imgPath,'numThreads',1,'resize',[Size,Size],'pack','SubtractAverage',net.meta.normalization.averageImage);

%     res = vl_simplenn_modified(net,gpuArray(img{1}),[],[],'mode','test');
%     PSD(:,idx) = squeeze(gather(vl_nnsoftmax(res(end).x)));
%     pred_score(:,idx) = imdb.anchors' * squeeze(gather(vl_nnsoftmax(res(end).x)));
%     pred_score(:,idx) = squeeze(gather(res(end).x));
    

    net.mode = 'test' ;
    inputs = {'data', gpuArray(img{1})} ;
    net.eval(inputs) ;
%     PSD(:,idx) = squeeze(gather((net.vars(p).value)));
%     pred_score(idx) = imdb.anchors' * squeeze(gather((net.vars(p).value)));
    PSD(:,idx) = squeeze(gather(vl_nnsoftmax(net.vars(p).value)));
    pred_score(idx) = imdb.anchors' * squeeze(gather(vl_nnsoftmax(net.vars(p).value)));

%     pred_score(idx) = squeeze(gather(vl_nnsigmoid(net.vars(p).value))); 
%     for ss = 1:bs
%         src_path = imgPath{ss};
%         dst_path = fullfile('databases/AVA_dataset/test',['g' num2str(groundtruth_score(idx(ss))) '_p' num2str(pred_score(idx(ss))) '.jpg']);
%         copyfile(src_path,dst_path);
%     end
end
pred_std = sum(bsxfun(@minus,pred_score',[1:10]).^2 .* PSD',2);
% load('pred_score_res512.mat','pred_score_res512');
% load('pred_score_res448.mat','pred_score');
% load('pred_score_g0_5_res384.mat','pred_score_g0_5_res384');
% load('pred_score_g0_5_res128.mat','pred_score_g0_5_res128');
% load('pred_score_g0_25.mat','pred_score_g0_25');
% pred_score = (pred_score_res512 + pred_score_g0_5_res384 + pred_score_g0_25)/3;

% load('pred_score_g0_5_vgg512.mat','pred_score_g0_5_vgg512');
% load('pred_score_g0_75_vgg384.mat','pred_score_g0_75_vgg384');
% load('pred_score_g0_25_vgg256.mat','pred_score_g0_25_vgg256');
% load('pred_score_512.mat','pred_score_512');
% load('pred_score_384.mat','pred_score_384');
% load('pred_score_256.mat','pred_score_256');
% pred_score = pred_score_g0_5_vgg512 + pred_score_g0_75_vgg384 + pred_score_g0_25_vgg256;
% pred_score = pred_score_512 + pred_score_g0_75_vgg384 +pred_score_256;
% pred_score = pred_score_g0_25_vgg256;

% load('pred_score_512.mat','pred_score_512');
% load('pred_score_vgg384.mat','pred_score_vgg384');
% load('pred_score_vgg256.mat','pred_score_vgg256');
% pred_score = (pred_score_vgg384 + pred_score_vgg256)/2;
% pred_score = pred_score_vgg384;

groundtruth_label = ones(1,numel(testID));
beta = 5;
groundtruth_label(groundtruth_score<=5) = -1;
pred_label = ones(1,numel(testID));
pred_label(pred_score<=beta) = -1;
acc = sum(pred_label==groundtruth_label) / numel(testID);
srcc = corr(pred_score',groundtruth_score','type','Spearman');
plcc = corr(pred_score',groundtruth_score','type','Pearson');
MAE = mean(abs(groundtruth_score-pred_score));
MSE = mean((groundtruth_score-pred_score).^2);

srcc_std = corr(pred_std,imdb.images.std(testID)','type','Spearman');
plcc_std = corr(pred_std,imdb.images.std(testID)','type','Pearson');
% EMD = mean(sqrt(mean((cumsum(RSD)-cumsum(PSD)).^2)));
% PSD = max(PSD,0);
% KLD = mean(sum((- RSD .* log((PSD + eps(1))./(RSD  + eps(1))))));


% cnt = 1;
% for t = 2:0.5:8
%     sel = find(groundtruth_score>=t & groundtruth_score<t+0.5);
%     acc(cnt) = sum(pred_label(sel)==groundtruth_label(sel)) / numel(sel);
%     srcc(cnt) = corr(pred_score(sel)',groundtruth_score(sel)','type','Spearman');
%     num(cnt) = numel(sel);
%     cnt = cnt + 1;
% end
% subplot(2,1,1)
% bar(acc)
% xticklabels({[2:0.5:8]});
% ylabel('acc');
% errnum = (1-acc) .* num;
% subplot(2,1,2)
% bar(errnum)
% xticklabels({[2:0.5:8]});
% ylabel('number of error samples');
