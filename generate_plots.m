% Submit your textual answers, and attach these plots in a latex file for
% this homework. 
% This script is merely for your convenience, to generate the plots for each
% experiment. Feel free to change it, as you do not need to submit this
% with your code.

% Loading the data: this loads X, and Ytrain.
load('/Users/sauravbose/Data Science/Machine Learning/HW2/hw2_release/data/X_noisy.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('/Users/sauravbose/Data Science/Machine Learning/HW2/hw2_release/data/Y.mat');

K = 1;
distfunc = 'l2';


N_folds = [2,4,8,16];
errors_xval = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the [N_folds(j)]-fold cross validation error in trial i 
errors_test = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the true test error in trial i (the entire row will be identical)

for trial = 1:100
    train_rows = randperm(length(X),450);
    X_train = X(train_rows,:);
    Y_train = Y(train_rows);
    
    test_rows = setdiff(1:length(X),train_rows);
    X_test = X(test_rows,:);
    Y_test = Y(test_rows);
    
    true_labels = k_nearest_neighbours(X_train,Y_train,X_test,K,distfunc)
    y_diff = true_labels - Y_test;
    y_err = y_diff(y_diff~=0);
    err = length(y_err)/length(y_diff);
    
    for j = 1:length(N_folds)
       part = make_xval_partition(length(Y_train), N_folds(j));
       errors_xval(trial,j) = knn_xval_error(X_train, Y_train, K, part, distFunc);
       
       errors_test(trial,j) = err;
               
    end
       
end

% code to plot the error bars. change these values depending on what
% experiment you are running
y = mean(errors_xval); e = std(errors_xval); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_test); e = std(errors_test); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, N = [2,4,8,16]');
xlabel('N');
ylabel('Error');
xlim([0,18]);
legend('N-Fold Error','Test Error');
hold off;