% Submit your textual answers, and attach these plots in a latex file for
% this homework. 
% This script is merely for your convenience, to generate the plots for each
% experiment. Feel free to change it, as you do not need to submit this
% with your code.

% Loading the data: this loads X, and Ytrain.
load('/Users/sauravbose/Data Science/Machine Learning/HW2/hw2_release/data/X_noisy.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('/Users/sauravbose/Data Science/Machine Learning/HW2/hw2_release/data/Y.mat');

%K = [1,2,3,5,8,13,21,34];
d_lim = 1:20;
%distfunc = 'l2';


N_folds = 10;
errors_xval = zeros(100,size(d_lim,2)); % here errors_xval(i,j) records the [N_folds(j)]-fold cross validation error in trial i 
errors_test = zeros(100,size(d_lim,2)); % here errors_xval(i,j) records the true test error in trial i (the entire row will be identical)

for trial = 1:100
    train_rows = randperm(length(X),450);
    X_train = X_noisy(train_rows,:);
    Y_train = Y(train_rows);
    
    test_rows = setdiff(1:length(X),train_rows);
    X_test = X_noisy(test_rows,:);
    Y_test = Y(test_rows);
    Y_test = double(Y_test);
    
    p = []
    part = make_xval_partition(length(Y_train), 10);
       
    for j = 1:length(d_lim) %here
        p=[];
       root = dt_train(X_train,Y_train,d_lim(j)); 
       for k = 1:length(X_test)
            p(k) = dt_value(root, X_test(k,:));
       end
       p = p';
       labels = p;
       labels(labels>=0.5)=1;
       labels(labels<0.5)=0;
        %labels = k_nearest_neighbours(X_train,Y_train,X_test,K(j),'l2')
    
       y_diff = labels - Y_test;
       y_err = y_diff(y_diff~=0);

       err = length(y_err)/length(y_diff);
    
       

       errors_xval(trial,j) = dt_xval_error(X_train, Y_train, d_lim(j), part);
       
       %errors_xval(trial,j) = knn_xval_error(X_train, Y_train, K(j), part, 'l2')
       errors_test(trial,j) = err;
               
    end
       
end

% code to plot the error bars. change these values depending on what
% experiment you are running
y = mean(errors_xval); e = std(errors_xval); x = 1:20; % here <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_test); e = std(errors_test); x = 1:20; % here <- computes mean across all trials
errorbar(x, y, e);
title('Noisy data, Depth limit = [1,2,3,...,20]');
xlabel('Depth limit');
ylabel('Error');
xlim([0,21]);
legend('10-Fold Error','Test Error');
hold off;