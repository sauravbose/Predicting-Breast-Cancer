function labels = kernel_regression(Xtrain,Ytrain,Xtest,sigma)

    % Function that implements kernel regression on the given data (binary classification)
    % Usage: labels = kernel_regression(Xtrain,Ytrain,Xtest)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % sigma : width of the (gaussian) kernel.
    % labels : return an M x 1 vector of predicted labels for testing data.
    

    
    % YOUR CODE GOES HERE
   
    N = size(Xtrain,1);
    P = size(Xtrain,2);
    M = size(Xtest, 1);
    
  
    X_train_mat = reshape(Xtrain', [1,P,N]);
    X_train_mat = repmat(X_train_mat, [M 1 1]);
    
    
    X_test_mat = repmat(Xtest,[1 1 N]);
    
    diff_mat = X_train_mat - X_test_mat;
    
    dist_mat = sqrt(sum(diff_mat.^2,2));
    kernel = exp(-dist_mat.^2/sigma^2);
    
    kernel = squeeze(kernel);
   
    Y_mat = repmat(Ytrain', [M 1 1]);
    
    labels = sum(kernel.*Y_mat,2)./sum(kernel,2);
    labels(labels >= 0.5) = 1;
    labels(labels < 0.5) = 0;
            
end
