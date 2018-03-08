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
            case {'Gauss_OV', 'Gauss_MV', 'RSD'}
                net.addLayer('myLoss', dagnn.myLoss('lossType',opts.lossType), {'fc','label'}, 'objective') ;
                switch opts.lossType
                    case 'CE'
                        net.meta.trainOpts.learningRate = logspace(-2, -3, opts.epoch);
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
    end
end
