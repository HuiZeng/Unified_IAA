classdef myLoss < dagnn.Loss
    
  properties
    lossType = 'CE'
  end
  
  methods
    function outputs = forward(obj, inputs, params)
        X = inputs{1};
        c = inputs{2};
        c = reshape(c,size(X));
        if size(X,3) == 1
            switch obj.lossType
                case 'CE'
                    X = vl_nnsigmoid(X);
                    Y = sum(squeeze(c.*log(X) + (1-c).*log(1-X)));
                case 'MSE'
                    Y = sum(squeeze((X - c).^2));
                case 'Huber'
                    delta = 1 / 9;
                    a = abs(squeeze(X - c));
                    y1 = sum(a(a<=delta).^2);
                    y2 = sum((a(a>delta)-0.5*delta)*delta);
                    Y = y1 + y2;
            end
        else
            switch obj.lossType
                case 'CE'
                    X = vl_nnsoftmax(X);
                    Y = sum(sum(squeeze(- c .* log(X./(c  + eps(1))))));
                case 'MSE'
                    Y = squeeze(sum(sum((X - c).^2)));
                case 'Huber'
                    delta = 1 / 9;
                    a = abs(squeeze(X - c));
                    y1 = sum(sum(a(a<=delta).^2));
                    y2 = sum(sum((a(a>delta)-0.5*delta)*delta));
                    Y = y1 + y2;
            end
        end
        outputs{1} = Y;
        n = obj.numAveraged ;
        m = n + size(inputs{1},4);
        obj.average = (n * obj.average + gather(outputs{1})) / m ;
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        X = inputs{1};
        c = inputs{2};
        c = reshape(c,size(X));
        if size(X,3) == 1
            switch obj.lossType
                case 'CE'
                    X = vl_nnsigmoid(X);
                    Y = X - c;
                case 'MSE'
                    Y = X - c;
                case 'Huber'
                    delta = 1 / 9;
                    a = X - c;
                    Y = X - c;
                    Y(a>delta) = delta;
                    Y(a<-delta) = - delta;
            end
        else
            switch obj.lossType
                case 'CE'
                    X = vl_nnsoftmax(X);
                    Y = X-c;
                case 'MSE'
                    Y = X - c;
                case 'Huber'
                    delta = 1 / 9;
                    a = X - c;
                    Y = X - c;
                    Y(a>delta) = delta;
                    Y(a<-delta) = - delta;
            end
        end
        derInputs = {Y, []};
        derParams = {};  
    end

    function obj = myLoss(varargin)
      obj.load(varargin) ;
    end
  end
end


