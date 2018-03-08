function net = model_intialization(opts)

if isempty(opts.network)
    if strcmp(opts.Model,'resnet50')
        netStruct = load(fullfile('data','PretrainedModels','imagenet-resnet-50-dag.mat')) ;
        net = dagnn.DagNN.loadobj(netStruct) ;
        net.removeLayer('fc1000');
        net.removeLayer('prob');
        fc_size = [1 1 2048 opts.outputDim];
        fc5Block = dagnn.Conv('size',fc_size,'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
        if strcmp(opts.database,'AADB')
            dropoutBlock = dagnn.DropOut('rate',0.5);
            net.addLayer('dropout',dropoutBlock,{'pool5'},{'pool5d'},{});
            net.addLayer('fc',fc5Block,{'pool5d'},{'fc'},{'fc_filter','fc_bias'});   
        else
            net.addLayer('fc',fc5Block,{'pool5'},{'fc'},{'fc_filter','fc_bias'}); 
        end
        p = net.getParamIndex(net.layers(end).params) ;
        params = net.layers(end).block.initParams() ;
        [net.params(p).value] = deal(params{:}) ;
        
        switch opts.labelType
            case 'binaryLabel'
                net.addLayer('logistic', dagnn.Loss('loss', 'logistic'), {'fc','label'}, 'objective') ; 
                net.meta.trainOpts.learningRate = logspace(-3, -4, opts.epoch);
            case 'qualityScore'
                net.addLayer('myLoss', dagnn.myLoss('lossType',opts.lossType), {'fc','label'}, 'objective') ;
                switch opts.lossType
                    case 'CE'
                        net.meta.trainOpts.learningRate = logspace(-2, -3, opts.epoch);
                    case 'MSE'
                        net.meta.trainOpts.learningRate = logspace(-2.5, -3.5, opts.epoch);
                    case 'Huber'
                        net.meta.trainOpts.learningRate = logspace(-1.5, -2.5, opts.epoch);
                end
                net.meta.trainOpts.learningRate = logspace(-2.5, -3.5, opts.epoch);
            case {'Gauss_OV', 'Gauss_MV', 'RSD'}
                net.addLayer('myLoss', dagnn.myLoss('lossType',opts.lossType), {'fc','label'}, 'objective') ;
                switch opts.lossType
                    case 'CE'
                        net.meta.trainOpts.learningRate = logspace(-3, -4, opts.epoch);
                    case 'MSE'
                        net.meta.trainOpts.learningRate = logspace(-2.5, -3.5, opts.epoch);
                    case 'Huber'
                        net.meta.trainOpts.learningRate = logspace(-1.5, -2.5, opts.epoch);
                end
        end
        p = net.getVarIndex('fc') ;
        net.vars(p).precious = 1;
        net.meta.trainOpts.weightDecay = 0.0001;
        net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate);
        net.meta.trainOpts.batchSize = 32;
        net.meta.trainOpts.numSubBatches = 1;
    elseif strcmp(opts.Model,'vgg16')
        net = load(fullfile('data','PretrainedModels','imagenet-vgg-verydeep-16.mat')); 
        net = vl_simplenn_tidy(net);
        net.layers = net.layers(1:31);
        net.layers{end+1} = struct('type', 'pool', 'method', 'avg', 'pool', [12 12], 'stride', 1, 'pad', 0) ;
        net.layers{end+1} = struct('type', 'conv', 'name', 'fc6',...
                                   'weights', {{randn(1, 1, 512, 4096, 'single')*0.01,zeros(1,4096,'single')}},...
                                   'learningRate', [1 1],'stride', 1, 'pad', 0);
        net.layers{end+1} = struct('type', 'relu','name', 'relu6');
        if strcmp(opts.database,'AADB')
            layer = struct('type', 'dropout','rate', 0.5) ;
            net.layers{end+1} = layer;
        end
        initialW = 0.01 * randn(1,1,4096,opts.outputDim,'single');
        initialBias = 0.1.*ones(1, opts.outputDim, 'single');
        net.layers{end+1} = struct('type', 'conv', 'name', 'classifier', ...
                'weights', {{initialW, initialBias}}, ...
                'stride', 1, 'pad', 0, 'learningRate', [1 1], 'weightDecay', [1 1]) ;

        switch opts.labelType
            case 'qualityScore'
                net.layers{end+1} = struct('type', 'softmax_regressionloss', 'precious',1) ;
                net.meta.trainOpts.learningRate = logspace(-2.5, -3.5, opts.epoch);
            case 'binaryLabel'
                net.layers{end+1} = struct('type', 'loss', 'precious',1) ;
                net.meta.trainOpts.learningRate = logspace(-3, -4, opts.epoch);
            case {'Gauss_OV', 'Gauss_MV', 'RSD'}
                net.layers{end+1} = struct('type', 'softmax_regressionloss', 'precious',1) ;
                net.meta.trainOpts.learningRate = logspace(-3, -4, opts.epoch);
        end
        net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
        net.meta.trainOpts.batchSize = 32 ;
        net.meta.trainOpts.numSubBatches = 1;
        net.meta.trainOpts.weightDecay = 0.0005;
    end
end