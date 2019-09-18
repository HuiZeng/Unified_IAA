function imdb = setup_AADB()

base_dir = fullfile('/home/zenghui/Documents/Image Aesthetics/databases/AADB');
Info = load(fullfile(base_dir,'AADBinfo2.mat'));
Info = Info.Info;

score = [Info.trainScore, Info.testScore]';
imageList = [Info.trainNameList Info.testNameList];
set = ones(1,numel(score));
set(end-999:end) = 3;
labels = ones(1,numel(score));
idx0 = find(score<=0.5);
labels(idx0) = -1;


AllScores = [Info.trainAllScores,Info.testAllScores];
for i = 1:numel(AllScores)
    meanScore(i,1) = mean(AllScores{i});
    ori_var(i,1) = std(AllScores{i})+1e-3;
    RSD(i,:) = hist(AllScores{i},[1:5])/length(AllScores{i});
end
mod_var = min(1.5,0.5*ori_var);
imdb.anchors = [1:0.5:5]'; % [1:5] for RSD [1:0.5:5] for Gauss
Gauss = bsxfun(@minus,meanScore,imdb.anchors');
Gauss_OV = exp(-bsxfun(@rdivide,(Gauss.^2),ori_var));
Gauss_OV = bsxfun(@rdivide,Gauss_OV,sum(Gauss_OV,2));
Gauss_MV = exp(-bsxfun(@rdivide,(Gauss.^2),mod_var));
Gauss_MV = bsxfun(@rdivide,Gauss_MV,sum(Gauss_MV,2));

sel = (score~=0);
imdb.images.score = meanScore(sel)';
imdb.images.labels = single(labels(sel));
imdb.images.set = set(sel);
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.base_dir = base_dir;
imdb.meta.imageList = imageList(sel);

imdb.images.RSD = RSD(sel,:)';
imdb.images.Gauss_OV = Gauss_OV(sel,:)';
imdb.images.Gauss_MV = Gauss_MV(sel,:)';


