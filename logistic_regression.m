function labels = logistic_regression(Xtrain,Ytrain,Xtest,stepsize,iterations)

    % Function to perform logistic regression on the given data (binary classification)
    % Usage: labels = logistic_regression(Xtrain,Ytrain,Xtest)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    % You may use gradient descent as a subroutine within this function to
    % make things simpler. Use the version of gradient descent
    % (constant step size, variable step size) and the step size which
    % you found to work best empirically.
    
    % Remember: Any modification you might wish to make on the
    % training & testing data must be done here (e.g. adding a new feature).
    % Remember: logistic regression will return probability values, and not
    % the actual labels themselves. This function has to return binary
    % labels, so you will have to perform some thresholding on the computed
    % probability values.
    
    stepsize = 0.0001;
    iterations = 500;
    col_new_train = ones(length(Xtrain),1);
    Xtrain = [Xtrain,col_new_train];
    
    col_new_test = ones(length(Xtest),1);
    Xtest = [Xtest,col_new_test];
    
    %[weights,error_per_iter] = gradient_ascent_decay(Xtrain,Ytrain,stepsize,iterations);
    [weights,error_per_iter] = gradient_ascent_fixed(Xtrain,Ytrain,stepsize,iterations)
    
     labels = 1./(1+exp(-Xtest*weights));
     labels(labels>=0.5)=1;
     labels(labels<0.5)=0;
    

end