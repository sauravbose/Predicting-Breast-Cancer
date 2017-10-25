function [error] = logistic_xval_error(X, Y, part)
% LOGISTIC_XVAL_ERROR - Logistic regression cross-validation error.
%
% Usage:
%
%   ERROR = logistic_xval_error(X, Y, PART)
%
% Returns the average N-fold cross validation error of the logistic regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, LOGISTIC_REGRESSION

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
     
     label = logistic_regression(X_train,Y_train,X_test,0.0001,500);
     
     y_diff = label - Y_test;
     y_err = y_diff(y_diff~=0);
     
     err = length(y_err)/length(y_diff);
     error_vect = cat(2,error_vect,err);
     
 end
 error = mean(error_vect);

end