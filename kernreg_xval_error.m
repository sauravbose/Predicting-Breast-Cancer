function [error] = kernreg_xval_error(X, Y, sigma, part)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(X, Y, SIGMA, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNEL_REGRESSION

% FILL IN YOUR CODE HERE

 N = max(part);
 
 error_vect = [];
 
 for i = 1:N
     test_rows = find(part == i);
     X_test = X(test_rows,:);
     Y_test = Y(test_rows);
     %Y_test = Y_test';
     train_rows = find(part~=i);
     X_train = X(train_rows,:);
     Y_train = Y(train_rows);
     %Y_train = Y_train';
     
     label = kernel_regression(X_train,Y_train,X_test,sigma);
     
     y_diff = label - Y_test;
     y_err = y_diff(y_diff~=0);
     
     err = length(y_err)/length(y_diff);
     error_vect = cat(2,error_vect,err);
     
 end
 error = mean(error_vect);
end 
 
 
 