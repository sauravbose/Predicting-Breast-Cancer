
train_rows = randperm(length(X),450);
X_train = X(train_rows,:);% change X to X_noisy for noisy data
Y_train = Y(train_rows);

test_rows = setdiff(1:length(X),train_rows);
X_test = X(test_rows,:); % Change X to X_noisy for noisy data
Y_test = Y(test_rows);

p = []
%labels_logist = logistic_regression(X_train,Y_train,X_test,0.0001,500)
%labels_knn = k_nearest_neighbours(X_train,Y_train,X_test,34,'l2')
%labels_kern = kernel_regression(X_train,Y_train,X_test,9);
root = dt_train(X_train,Y_train,6);
for i = 1:length(X_test)
p(i) = dt_value(root, X_test(i,:));
end
p = p';
labels = p;
labels(labels>=0.5)=1;
labels(labels<0.5)=0;

y_diff = labels - Y_test;
y_err = y_diff(y_diff~=0);
     
err = length(y_err)/length(y_diff);

