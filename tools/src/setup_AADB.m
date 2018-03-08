function imdb = setup_AADB()

base_dir = fullfile('databases','AADB');
Info = load(fullfile(base_dir,'AADBinfo.mat'));


score = [Info.trainScore, Info.testScore];
imageList = [Info.trainNameList Info.testNameList];
set = ones(1,numel(score));
set(end-999:end) = 3;
labels = ones(1,numel(score));
idx0 = find(score<=0.5);
labels(idx0) = -1;

imdb.images.score = score;
imdb.images.labels = single(labels);
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.base_dir = base_dir;
imdb.meta.imageList = imageList;



