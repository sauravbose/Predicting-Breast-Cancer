function labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K,distfunc)

    % Function to implement the K nearest neighbours algorithm on the given
    % dataset.
    % Usage: labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % K : number of nearest neighbours used to make predictions on the test
    %     dataset. Remember to take care of corner cases.
    % distfunc: distance function to be used - l1, l2, linf.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    % YOUR CODE GOES HERE.
    
    if nargin<5
        distfunc = 'l2';
    end
    
    testPointsNum = size(Xtest, 1);
    trainPointsNum = size(Xtrain, 1);
    
    trainVect = reshape(Xtrain', [1 size(Xtrain,2) trainPointsNum]);
    trainComparisonVect = repmat(trainVect, [testPointsNum 1 1]);
    testComparisonVect = repmat(Xtest, [1 1 trainPointsNum]);
    
    differenceVect = testComparisonVect - trainComparisonVect;
    
    if strcmp(distfunc, 'l2')
        distanceVect = sqrt(sum(differenceVect.^2, 2));
    elseif strcmp(distfunc, 'l1')
        distanceVect = sum(abs(differenceVect), 2);
    elseif strcmp(distfunc, 'linf')
        distanceVect = max(abs(differenceVect), [], 2);
    else
        error('Unrecognized distance function');
    end
    
    distanceVect = squeeze(distanceVect);
    
    if testPointsNum == 1 
        distanceVect = distanceVect';
    end

    [sorted, NNVect] = sort(distanceVect, 2);

    NNVect = NNVect(:,1:K);
    
    labels = Ytrain(NNVect);
    
    if size(NNVect, 1) == 1 
        labels = labels';
    end
    
    if K > 1
        labels = mean(labels, 2);
        labels(labels >= 0.5) = 1;
        labels(labels < 0.5) = 0;
    end
end