function imdb = setup_AVA()

base_dir = fullfile('databases/AVA_dataset');
Info = load(fullfile(base_dir,'AVA.txt'));
test_file = load(fullfile(base_dir,'aesthetics_image_lists','generic_test.jpgl'));
ID = 1:size(Info);
[train_file,trainID] = setdiff(Info(:,2),test_file);
testID = setdiff(ID,trainID);
score_matrix = Info(:,3:12);

Number = sum(score_matrix,2);
RSD = bsxfun(@rdivide,score_matrix,Number);
imdb.anchors = [1:10]';
score = RSD * imdb.anchors;
ori_var = sum(bsxfun(@minus,score,[1:10]).^2 .* RSD,2);
mod_var = min(1.5,0.5*ori_var);

Gauss = bsxfun(@minus,score,imdb.anchors');
Gauss_OV = exp(-bsxfun(@rdivide,(Gauss.^2),ori_var));
Gauss_OV = bsxfun(@rdivide,Gauss_OV,sum(Gauss_OV,2));
Gauss_MV = exp(-bsxfun(@rdivide,(Gauss.^2),mod_var));
Gauss_MV = bsxfun(@rdivide,Gauss_MV,sum(Gauss_MV,2));

MARE = mean(abs(Gauss_MV * [1:10]' - score))


set = ones(1,numel(score));
set(testID) = 3;
for i = 1:size(Info,1)
    imageList{i} = [num2str(Info(i,2)) '.jpg'];
end
labels = ones(1,numel(score));
idx0 = find(score<=5);
labels(idx0) = -1;
    
if ~exist(fullfile('data','AVA','image_exist_flag.mat'),'file')
    for i = 1:numel(imageList)
        fprintf('Finished image %d / %d...\n', i, numel(imageList));
        img=vl_imreadjpeg({fullfile(base_dir, 'images', imageList{i})});
        if ~isempty(img{1})
            flag(i) = true;
        else
            flag(i) = false;
        end
    end
    save(fullfile('data','AVA','image_exist_flag.mat'),'flag');
else
    load(fullfile('data','AVA','image_exist_flag.mat'),'flag');
end


imdb.images.score = score(flag)';
imdb.images.labels = single(labels(flag));

imdb.images.RSD = RSD(flag,:)';
imdb.images.Gauss_OV = Gauss_OV(flag,:)';
imdb.images.Gauss_MV = Gauss_MV(flag,:)';

imdb.images.set = set(flag);
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.base_dir = base_dir;
imdb.meta.imageList = imageList(flag);


imdb.images.std = sqrt(ori_var(flag)');
% imdb.images.std = (imdb.images.std - min(imdb.images.std))/ (max(imdb.images.std) - min(imdb.images.std)) * 9.99;
% imdb.images.score = imdb.images.std;


